//! SPIR-T passes related to debuginfo.

use crate::custom_decorations::{CustomDecoration, SrcLocDecoration, ZombieDecoration};
use crate::custom_insts::{self, CustomInst, CustomOp};
use itertools::Either;
use rustc_data_structures::fx::{FxHashMap, FxIndexSet};
use smallvec::SmallVec;
use spirt::func_at::FuncAtMut;
use spirt::transform::{InnerInPlaceTransform as _, Transformed, Transformer};
use spirt::visit::InnerVisit;
use spirt::{
    spv, Attr, AttrSet, AttrSetDef, Const, ConstKind, Context, ControlNodeKind, ControlRegion,
    DataInstForm, DataInstKind, DbgSrcLoc, Diag, InternedStr, Module, Type, Value,
};
use std::marker::PhantomData;
use std::str;

/// Replace our custom decorations ([`SrcLocDecoration`] and [`ZombieDecoration`])
/// and custom extended instruction debuginfo, with SPIR-T equivalents.
//
// FIXME(eddyb) rename this module because it's not just about debuginfo.
pub fn convert_custom_decorations_and_debuginfo_to_spirt(
    linker_options: &crate::linker::Options,
    module: &mut Module,
) {
    let cx = &module.cx();

    // FIXME(eddyb) reuse this collection work in some kind of "pass manager".
    let (all_global_vars, all_funcs) = {
        let mut collector = super::ReachableUseCollector {
            cx,
            module,

            seen_types: FxIndexSet::default(),
            seen_consts: FxIndexSet::default(),
            seen_data_inst_forms: FxIndexSet::default(),
            seen_global_vars: FxIndexSet::default(),
            seen_funcs: FxIndexSet::default(),
        };
        for (export_key, &exportee) in &module.exports {
            export_key.inner_visit_with(&mut collector);
            exportee.inner_visit_with(&mut collector);
        }
        (collector.seen_global_vars, collector.seen_funcs)
    };

    let mut transformer = CustomDecorationsAndDebuginfoToSpirt {
        linker_options,
        cx,
        custom_ext_inst_set: cx.intern(&custom_insts::CUSTOM_EXT_INST_SET[..]),
        transformed_types: FxHashMap::default(),
        transformed_consts: FxHashMap::default(),
        transformed_data_inst_forms: FxHashMap::default(),
    };
    for gv in all_global_vars {
        transformer.in_place_transform_global_var_decl(&mut module.global_vars[gv]);
    }
    for func in all_funcs {
        transformer.in_place_transform_func_decl(&mut module.funcs[func]);
    }
}

// HACK(eddyb) version of `decorations::LazilyDecoded` that works for SPIR-T.
struct LazilyDecoded<D> {
    encoded: String,
    _marker: PhantomData<D>,
}

impl<D> LazilyDecoded<D> {
    fn try_from_prefixed<'a>(prefixed_encoded: &str) -> Option<Self>
    where
        D: CustomDecoration<'a>,
    {
        let encoded = prefixed_encoded.strip_prefix(D::ENCODING_PREFIX)?;

        Some(Self {
            encoded: encoded.to_string(),
            _marker: PhantomData,
        })
    }

    fn decode<'a>(&'a self) -> D
    where
        D: CustomDecoration<'a>,
    {
        D::decode(&self.encoded)
    }
}

fn decode_all_custom_decorations(
    attrs_def: &AttrSetDef,
) -> impl Iterator<
    Item = Either<LazilyDecoded<SrcLocDecoration<'_>>, LazilyDecoded<ZombieDecoration<'_>>>,
> + '_ {
    let wk = &super::SpvSpecWithExtras::get().well_known;

    attrs_def.attrs.iter().filter_map(|attr| {
        let spv_inst = match attr {
            Attr::SpvAnnotation(spv_inst) if spv_inst.opcode == wk.OpDecorateString => spv_inst,
            _ => return None,
        };
        let str_imms = spv_inst
            .imms
            .strip_prefix(&[spv::Imm::Short(wk.Decoration, wk.UserTypeGOOGLE)])?;

        super::decode_spv_lit_str_with(str_imms, |prefixed_encoded| {
            Option::or(
                LazilyDecoded::<SrcLocDecoration<'_>>::try_from_prefixed(prefixed_encoded)
                    .map(Either::Left),
                LazilyDecoded::<ZombieDecoration<'_>>::try_from_prefixed(prefixed_encoded)
                    .map(Either::Right),
            )
        })
    })
}

struct CustomDecorationsAndDebuginfoToSpirt<'a> {
    linker_options: &'a crate::linker::Options,

    cx: &'a Context,

    /// Interned name for our custom "extended instruction set"
    /// (see `crate::custom_insts` for more details).
    custom_ext_inst_set: InternedStr,

    // FIXME(eddyb) build some automation to avoid ever repeating these.
    transformed_types: FxHashMap<Type, Transformed<Type>>,
    transformed_consts: FxHashMap<Const, Transformed<Const>>,
    transformed_data_inst_forms: FxHashMap<DataInstForm, Transformed<DataInstForm>>,
}

impl Transformer for CustomDecorationsAndDebuginfoToSpirt<'_> {
    // FIXME(eddyb) build some automation to avoid ever repeating these.
    fn transform_type_use(&mut self, ty: Type) -> Transformed<Type> {
        if let Some(&cached) = self.transformed_types.get(&ty) {
            return cached;
        }
        let transformed = self
            .transform_type_def(&self.cx[ty])
            .map(|ty_def| self.cx.intern(ty_def));
        self.transformed_types.insert(ty, transformed);
        transformed
    }
    fn transform_const_use(&mut self, ct: Const) -> Transformed<Const> {
        if let Some(&cached) = self.transformed_consts.get(&ct) {
            return cached;
        }
        let transformed = self
            .transform_const_def(&self.cx[ct])
            .map(|ct_def| self.cx.intern(ct_def));
        self.transformed_consts.insert(ct, transformed);
        transformed
    }
    fn transform_data_inst_form_use(
        &mut self,
        data_inst_form: DataInstForm,
    ) -> Transformed<DataInstForm> {
        if let Some(&cached) = self.transformed_data_inst_forms.get(&data_inst_form) {
            return cached;
        }
        let transformed = self
            .transform_data_inst_form_def(&self.cx[data_inst_form])
            .map(|data_inst_form_def| self.cx.intern(data_inst_form_def));
        self.transformed_data_inst_forms
            .insert(data_inst_form, transformed);
        transformed
    }

    fn transform_attr_set_use(&mut self, attrs: AttrSet) -> Transformed<AttrSet> {
        let orig_attrs = attrs;
        let mut attrs = attrs;
        for src_loc_or_zombie in decode_all_custom_decorations(&self.cx[attrs]) {
            let src_loc_or_zombie = src_loc_or_zombie
                .as_ref()
                .map_either(|src_loc| src_loc.decode(), |zombie| zombie.decode());
            match src_loc_or_zombie {
                Either::Left(SrcLocDecoration {
                    file_name,
                    line_start,
                    line_end,
                    col_start,
                    col_end,
                }) => attrs.set_dbg_src_loc(
                    self.cx,
                    DbgSrcLoc {
                        file_path: self.cx.intern(file_name),
                        start_line_col: (line_start, col_start),
                        end_line_col: (line_end, col_end),
                        inlined_callee_name_and_call_site: None,
                    },
                ),
                Either::Right(ZombieDecoration { reason }) => {
                    let diag = if self.linker_options.early_report_zombies {
                        Diag::bug([
                            format!("zombie should've been reported already: {reason}").into()
                        ])
                    } else {
                        Diag::err([reason.to_string().into()])
                    };
                    attrs.push_diag(self.cx, diag);
                }
            }
        }
        if attrs != orig_attrs {
            Transformed::Changed(attrs)
        } else {
            Transformed::Unchanged
        }
    }

    fn in_place_transform_control_region_def(
        &mut self,
        mut func_at_region: FuncAtMut<'_, ControlRegion>,
    ) {
        // HACK(eddyb) buffering the `DataInst`s to remove from this region's blocks,
        // as iterating and modifying a list at the same time isn't supported.
        let mut insts_to_remove = SmallVec::<[_; 8]>::new();

        // HACK(eddyb) this relies on the fact that each original SPIR-V block
        // can't be broken up into separate regions (at least for now), which
        // may not necessarily always remain true, and steps should be taken
        // elsewhere to explicitly unset debuginfo, instead of relying on the
        // end of region implicitly unsetting it all (without too much leakage).
        let mut dbg_src_loc = None;

        // HACK(eddyb) only needed becasue `dbg_src_loc` can be `None`, but when
        // `dbg_src_loc` is `Some`, its `inlined_callee_name_and_call_site` field
        // must have the same value as this variable.
        let mut inlined_callee_name_and_call_site = None;

        let mut children = func_at_region.reborrow().at_children().into_iter();
        while let Some(mut func_at_control_node) = children.next() {
            let control_node = func_at_control_node.position;

            // HACK(eddyb) flatten only `Block`s (into their `DataInst`s).
            let mut func_at_insts_or_node = match func_at_control_node.reborrow().def().kind {
                ControlNodeKind::Block { insts } => {
                    Either::Left(func_at_control_node.at(insts).into_iter())
                }
                _ => Either::Right([func_at_control_node].into_iter()),
            };
            loop {
                let Some(mut func_at_inst_or_node) = func_at_insts_or_node
                    .as_mut()
                    .map_either(|it| it.next(), |it| it.next())
                    .factor_none()
                else {
                    break;
                };

                // FIXME(eddyb) deduplicate with `spirt_passes::diagnostics`.
                let maybe_custom_inst =
                    func_at_inst_or_node
                        .as_mut()
                        .left()
                        .and_then(|func_at_inst| {
                            let data_inst_def = func_at_inst.reborrow().freeze().def();
                            match self.cx[data_inst_def.form].kind {
                                DataInstKind::SpvExtInst {
                                    ext_set,
                                    inst: ext_inst,
                                    lowering: _,
                                } if ext_set == self.custom_ext_inst_set => Some(
                                    CustomOp::decode(ext_inst).with_operands(&data_inst_def.inputs),
                                ),
                                _ => None,
                            }
                        });
                if let Some(custom_inst) = maybe_custom_inst {
                    let func_at_inst = func_at_inst_or_node.as_mut().left().unwrap().reborrow();
                    let inst = func_at_inst.position;
                    let mut mark_for_removal = || insts_to_remove.push((control_node, inst));
                    let expect_const = |v| match v {
                        Value::Const(ct) => ct,
                        _ => unreachable!(),
                    };
                    let const_str = |v| match self.cx[expect_const(v)].kind {
                        ConstKind::SpvStringLiteralForExtInst(s) => s,
                        _ => unreachable!(),
                    };
                    let const_u32 = |v| {
                        expect_const(v)
                            .as_scalar(self.cx)
                            .unwrap()
                            .int_as_u32()
                            .unwrap()
                    };
                    match custom_inst {
                        CustomInst::SetDebugSrcLoc {
                            file,
                            line_start,
                            line_end,
                            col_start,
                            col_end,
                        } => {
                            dbg_src_loc = Some(DbgSrcLoc {
                                file_path: const_str(file),
                                start_line_col: (const_u32(line_start), const_u32(col_start)),
                                end_line_col: (const_u32(line_end), const_u32(col_end)),
                                inlined_callee_name_and_call_site,
                            });
                            mark_for_removal();
                            continue;
                        }
                        CustomInst::ClearDebugSrcLoc => {
                            dbg_src_loc = None;
                            mark_for_removal();
                            continue;
                        }
                        CustomInst::PushInlinedCallFrame { callee_name } => {
                            let mut call_site_attrs = AttrSet::default();
                            let mut call_site_loc = dbg_src_loc.take();

                            // HACK(eddyb) this accounts for the same situation
                            // described later below (missing `SetDebugSrcLoc`),
                            // but between consecutive `PushInlinedCallFrame`s.
                            if inlined_callee_name_and_call_site.is_some() {
                                call_site_loc
                                    .get_or_insert_with(|| DbgSrcLoc {
                                        file_path: self.cx.intern(""),
                                        start_line_col: (0, 0),
                                        end_line_col: (0, 0),
                                        inlined_callee_name_and_call_site: None,
                                    })
                                    .inlined_callee_name_and_call_site =
                                    inlined_callee_name_and_call_site;
                            }

                            if let Some(call_site_loc) = call_site_loc {
                                call_site_attrs.set_dbg_src_loc(self.cx, call_site_loc);
                            }
                            inlined_callee_name_and_call_site =
                                Some((const_str(callee_name), call_site_attrs));

                            // HACK(eddyb) work around missing `SetDebugSrcLoc`
                            // in the callee, by continuing to use the callsite
                            // for anything in the callee before `SetDebugSrcLoc`.
                            dbg_src_loc = call_site_loc;

                            mark_for_removal();
                            continue;
                        }
                        CustomInst::PopInlinedCallFrame => {
                            if let Some((_, call_site_attrs)) = inlined_callee_name_and_call_site {
                                let call_site_loc = call_site_attrs.dbg_src_loc(self.cx);
                                (dbg_src_loc, inlined_callee_name_and_call_site) = (
                                    call_site_loc,
                                    call_site_loc.and_then(|call_site_loc| {
                                        call_site_loc.inlined_callee_name_and_call_site
                                    }),
                                );
                                mark_for_removal();
                                continue;
                            } else {
                                func_at_inst.def().attrs.push_diag(
                                    self.cx,
                                    Diag::bug([
                                        "`PopInlinedCallFrame` without matching `PushInlinedCallFrame`"
                                            .into()
                                    ]),
                                );
                            }
                        }
                        CustomInst::Abort { .. } => {
                            let custom_op = custom_inst.op();
                            assert!(
                                !custom_op.is_debuginfo(),
                                "`CustomOp::{custom_op:?}` debuginfo not lowered"
                            );
                        }
                    }
                }

                let attrs = func_at_inst_or_node
                    .either(|fai| &mut fai.def().attrs, |facn| &mut facn.def().attrs);

                // Set the equivalent `Attr::DbgSrcLoc` attribute.
                if let Some(dbg_src_loc) = dbg_src_loc {
                    attrs.set_dbg_src_loc(self.cx, dbg_src_loc);
                }
            }
        }

        // Finally remove the `DataInst`s buffered for removal earlier.
        let func = func_at_region.reborrow().at(());
        for (parent_block, inst) in insts_to_remove {
            match &mut func.control_nodes[parent_block].kind {
                ControlNodeKind::Block { insts } => insts.remove(inst, func.data_insts),
                _ => unreachable!(),
            }
        }

        func_at_region.inner_in_place_transform_with(self);
    }
}
