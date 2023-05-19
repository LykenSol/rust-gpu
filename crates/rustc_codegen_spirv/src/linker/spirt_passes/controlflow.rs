//! SPIR-T passes related to control-flow.

use crate::custom_insts::{self, CustomInst, CustomOp};
use smallvec::SmallVec;
use spirt::func_at::FuncAt;
use spirt::{
    cfg, scalar, spv, Attr, AttrSet, ConstDef, ConstKind, ControlNodeKind, DataInstFormDef,
    DataInstKind, DbgSrcLoc, DeclDef, EntityDefs, ExportKey, Exportee, Module, Type, TypeKind,
    Value,
};
use std::fmt::Write as _;

/// Replace our custom extended instruction `Abort`s with standard `OpReturn`s,
/// but only in entry-points (and only before CFG structurization).
//
// FIXME(eddyb) no longer relying on structurization, try porting this
// to replace custom aborts in `Block`s and inject `ExitInvocation`s
// after them (truncating the `Block` and/or parent region if necessary).
pub fn convert_custom_aborts_to_unstructured_returns_in_entry_points(
    linker_options: &crate::linker::Options,
    module: &mut Module,
) {
    // HACK(eddyb) this shouldn't be the place to parse `abort_strategy`.
    enum Strategy {
        Unreachable,
        DebugPrintf { inputs: bool, backtrace: bool },
    }
    let abort_strategy = linker_options.abort_strategy.as_ref().map(|s| {
        if s == "unreachable" {
            return Strategy::Unreachable;
        }
        if let Some(s) = s.strip_prefix("debug-printf") {
            let (inputs, s) = s.strip_prefix("+inputs").map_or((false, s), |s| (true, s));
            let (backtrace, s) = s
                .strip_prefix("+backtrace")
                .map_or((false, s), |s| (true, s));
            if s.is_empty() {
                return Strategy::DebugPrintf { inputs, backtrace };
            }
        }
        panic!("unknown `--abort-strategy={s}");
    });

    let cx = &module.cx();
    let wk = &super::SpvSpecWithExtras::get().well_known;

    // HACK(eddyb) deduplicate with `diagnostics`.
    let name_from_attrs = |attrs: AttrSet| {
        cx[attrs].attrs.iter().find_map(|attr| match attr {
            Attr::SpvAnnotation(spv_inst) if spv_inst.opcode == wk.OpName => {
                Some(super::decode_spv_lit_str_with(&spv_inst.imms, |name| {
                    name.to_string()
                }))
            }
            _ => None,
        })
    };

    let custom_ext_inst_set = cx.intern(&custom_insts::CUSTOM_EXT_INST_SET[..]);

    for (export_key, exportee) in &module.exports {
        let (entry_point_imms, interface_global_vars, func) = match (export_key, exportee) {
            (
                ExportKey::SpvEntryPoint {
                    imms,
                    interface_global_vars,
                },
                &Exportee::Func(func),
            ) => (imms, interface_global_vars, func),
            _ => continue,
        };

        let func_decl = &mut module.funcs[func];
        assert!(func_decl.ret_types.is_empty());

        let func_def_body = match &mut func_decl.def {
            DeclDef::Present(def) => def,
            DeclDef::Imported(_) => continue,
        };

        let debug_printf_context_fmt_str;
        let mut debug_printf_context_inputs = SmallVec::<[_; 4]>::new();
        if let Some(Strategy::DebugPrintf { inputs, .. }) = abort_strategy {
            let mut fmt = String::new();

            match entry_point_imms[..] {
                [spv::Imm::Short(em_kind, _), ref name_imms @ ..] => {
                    assert_eq!(em_kind, wk.ExecutionModel);
                    super::decode_spv_lit_str_with(name_imms, |name| {
                        fmt += &name.replace('%', "%%");
                    });
                }
                _ => unreachable!(),
            }
            fmt += "(";

            // Collect entry-point inputs `OpLoad`ed by the entry block.
            // HACK(eddyb) this relies on Rust-GPU always eagerly loading inputs.
            let loaded_inputs = func_def_body
                .at(func_def_body
                    .at_body()
                    .at_children()
                    .into_iter()
                    .next()
                    .and_then(|func_at_first_node| match func_at_first_node.def().kind {
                        ControlNodeKind::Block { insts } => Some(insts),
                        _ => None,
                    })
                    .unwrap_or_default())
                .into_iter()
                .filter_map(|func_at_inst| {
                    let data_inst_def = func_at_inst.def();
                    let data_inst_form_def = &cx[data_inst_def.form];
                    if let DataInstKind::SpvInst(spv_inst, lowering) = &data_inst_form_def.kind {
                        if spv_inst.opcode == wk.OpLoad && lowering.disaggregated_output.is_none() {
                            if let Value::Const(ct) = data_inst_def.inputs[0] {
                                if let ConstKind::PtrToGlobalVar(gv) = cx[ct].kind {
                                    if interface_global_vars.contains(&gv) {
                                        return Some((
                                            gv,
                                            data_inst_form_def.output_types[0],
                                            Value::DataInstOutput {
                                                inst: func_at_inst.position,
                                                output_idx: 0,
                                            },
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    None
                });
            if inputs {
                let mut first_input = true;
                for (gv, ty, value) in loaded_inputs {
                    let vector_or_scalar_fmt = |ty: Type| {
                        let (scalar_type, vlen) = match cx[ty].kind {
                            TypeKind::Scalar(ty) => (ty, None),
                            TypeKind::Vector(ty) if (2..=4).contains(&ty.elem_count.get()) => {
                                (ty.elem, Some(ty.elem_count))
                            }
                            _ => return None,
                        };
                        let scalar_fmt = match scalar_type {
                            scalar::Type::S32 => "i",
                            scalar::Type::U32 => "u",
                            scalar::Type::F32 => "f",
                            _ => return None,
                        };
                        Some((scalar_fmt, vlen))
                    };
                    if let Some((scalar_fmt, vlen)) = vector_or_scalar_fmt(ty) {
                        if !first_input {
                            fmt += ", ";
                        }
                        first_input = false;

                        if let Some(name) = name_from_attrs(module.global_vars[gv].attrs) {
                            fmt += &name.replace('%', "%%");
                            fmt += " = ";
                        }
                        match vlen {
                            Some(vlen) => write!(fmt, "vec{vlen}(%v{vlen}{scalar_fmt})").unwrap(),
                            None => write!(fmt, "%{scalar_fmt}").unwrap(),
                        }
                        debug_printf_context_inputs.push(value);
                    }
                }
            }

            fmt += ")";

            debug_printf_context_fmt_str = fmt;
        } else {
            debug_printf_context_fmt_str = String::new();
        }

        let rpo_regions = func_def_body
            .unstructured_cfg
            .as_ref()
            .expect("Abort->OpReturn can only be done on unstructured CFGs")
            .rev_post_order(func_def_body);
        for region in rpo_regions {
            let region_def = &func_def_body.control_regions[region];
            let control_node_def = match region_def.children.iter().last {
                Some(last_node) => &mut func_def_body.control_nodes[last_node],
                _ => continue,
            };
            let block_insts = match &mut control_node_def.kind {
                ControlNodeKind::Block { insts } => insts,
                _ => continue,
            };

            let terminator = &mut func_def_body
                .unstructured_cfg
                .as_mut()
                .unwrap()
                .control_inst_on_exit_from[region];
            match terminator.kind {
                cfg::ControlInstKind::Unreachable => {}
                _ => continue,
            }

            // HACK(eddyb) this allows accessing the `DataInst` iterator while
            // mutably borrowing other parts of `FuncDefBody`.
            let func_at_block_insts = FuncAt {
                control_nodes: &EntityDefs::new(),
                control_regions: &EntityDefs::new(),
                data_insts: &func_def_body.data_insts,

                position: *block_insts,
            };
            let custom_terminator_inst = func_at_block_insts
                .into_iter()
                .next_back()
                .and_then(|func_at_inst| {
                    let data_inst_def = func_at_inst.def();
                    match cx[data_inst_def.form].kind {
                        DataInstKind::SpvExtInst {
                            ext_set,
                            inst,
                            lowering: _,
                        } if ext_set == custom_ext_inst_set => Some((
                            func_at_inst,
                            CustomOp::decode(inst).with_operands(&data_inst_def.inputs),
                        )),
                        _ => None,
                    }
                })
                .filter(|(_, custom)| custom.op().is_terminator());
            if let Some((
                func_at_abort_inst,
                CustomInst::Abort {
                    kind: abort_kind,
                    message_debug_printf,
                },
            )) = custom_terminator_inst
            {
                let abort_inst = func_at_abort_inst.position;
                terminator.kind = cfg::ControlInstKind::ExitInvocation(
                    cfg::ExitInvocationKind::SpvInst(wk.OpReturn.into()),
                );

                match abort_strategy {
                    Some(Strategy::Unreachable) => {
                        terminator.kind = cfg::ControlInstKind::Unreachable;
                    }
                    Some(Strategy::DebugPrintf {
                        inputs: _,
                        backtrace,
                    }) => {
                        let expect_const = |v| match v {
                            Value::Const(ct) => ct,
                            _ => unreachable!(),
                        };
                        let const_str = |v| match cx[expect_const(v)].kind {
                            ConstKind::SpvStringLiteralForExtInst(s) => s,
                            _ => unreachable!(),
                        };
                        let mk_const_str = |s| {
                            cx.intern(ConstDef {
                                attrs: Default::default(),
                                ty: cx.intern(TypeKind::SpvStringLiteralForExtInst),
                                kind: ConstKind::SpvStringLiteralForExtInst(s),
                            })
                        };

                        let mut fmt = String::new();

                        let (message_debug_printf_fmt_str, message_debug_printf_args) =
                            message_debug_printf
                                .split_first()
                                .map(|(&fmt_str, args)| (&cx[const_str(fmt_str)], args))
                                .unwrap_or_default();

                        let fmt_dbg_src_loc = |dbg_src_loc: DbgSrcLoc| {
                            let file = &cx[dbg_src_loc.file_path];
                            let (line, col) = dbg_src_loc.start_line_col;

                            // FIXME(eddyb) figure out what is going on with
                            // these column number conventions, below is a
                            // related comment from `spirt::print`:
                            // > // HACK(eddyb) Rust-GPU's column numbers seem
                            // > // off-by-one wrt what e.g. VSCode expects
                            // > // for `:line:col` syntax, but it's hard to
                            // > // tell from the spec and `glslang` doesn't
                            // > // even emit column numbers at all!
                            let col = col + 1;
                            format!("{file}:{line}:{col}").replace('%', "%%")
                        };

                        // HACK(eddyb) this improves readability w/ very verbose Vulkan loggers.
                        fmt += "\n";

                        fmt += "[Rust ";

                        // HACK(eddyb) turn "panic" into "panicked", while the
                        // general case looks like "abort" -> "aborted".
                        match &cx[const_str(abort_kind)] {
                            "panic" => fmt += "panicked",
                            verb => {
                                fmt += verb;
                                fmt += "en";
                            }
                        };

                        let mut dbg_src_loc = func_at_abort_inst.def().attrs.dbg_src_loc(cx);

                        if let Some(loc) = dbg_src_loc {
                            fmt += " at ";
                            fmt += &fmt_dbg_src_loc(loc);
                        }

                        fmt += "]\n ";
                        fmt += &message_debug_printf_fmt_str.replace('\n', "\n ");

                        let mut innermost = true;
                        let mut append_call =
                            |callee_name: &str, call_site_loc, callee_loc: Option<_>| {
                                if innermost {
                                    innermost = false;
                                    fmt += "\n      in ";
                                } else if callee_loc.is_some() {
                                    fmt += "\n      by ";
                                } else {
                                    // HACK(eddyb) previous call didn't have a `called at` line.
                                    fmt += "\n      called by ";
                                }
                                fmt += callee_name;
                                if let Some(loc) = call_site_loc {
                                    fmt += "\n        called at ";
                                    fmt += &fmt_dbg_src_loc(loc);
                                }
                            };
                        if backtrace {
                            while let Some((callee_name, call_site_attrs)) =
                                dbg_src_loc.and_then(|loc| loc.inlined_callee_name_and_call_site)
                            {
                                let call_site_loc = call_site_attrs.dbg_src_loc(cx);
                                append_call(
                                    &cx[callee_name].replace('%', "%%"),
                                    call_site_loc,
                                    dbg_src_loc,
                                );
                                dbg_src_loc = call_site_loc;
                            }
                        }
                        append_call(&debug_printf_context_fmt_str, None, dbg_src_loc);

                        fmt += "\n";

                        let abort_inst_def = &mut func_def_body.data_insts[abort_inst];
                        abort_inst_def.form = cx.intern(DataInstFormDef {
                            kind: DataInstKind::SpvExtInst {
                                ext_set: cx.intern("NonSemantic.DebugPrintf"),
                                inst: 1,
                                lowering: Default::default(),
                            },
                            output_types: cx[abort_inst_def.form].output_types.clone(),
                        });
                        abort_inst_def.inputs = [Value::Const(mk_const_str(cx.intern(fmt)))]
                            .into_iter()
                            .chain(message_debug_printf_args.iter().copied())
                            .chain(debug_printf_context_inputs.iter().copied())
                            .collect();

                        // Avoid removing the instruction we just replaced.
                        continue;
                    }
                    None => {}
                }
                block_insts.remove(abort_inst, &mut func_def_body.data_insts);
            }
        }
    }
}
