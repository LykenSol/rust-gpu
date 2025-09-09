// HACK(eddyb) avoids rewriting all of the imports (see `lib.rs` and `build.rs`).
use crate::maybe_pqp_cg_ssa as rustc_codegen_ssa;

// FIXME(eddyb) this should be implemented once for all backends, it doesn't
// need backend-specific logic, just a way to create modules and functions.

use crate::builder::Builder;
use crate::builder_spirv::{SpirvConst, SpirvFunctionCursor, SpirvValue};
use crate::codegen_cx::CodegenCx;
use crate::spirv_type::SpirvType;
use rspirv::spirv::{FunctionControl, LinkageType, Word};
use rustc_abi::HasDataLayout as _;
use rustc_ast::expand::allocator::{
    AllocatorMethod, AllocatorTy, NO_ALLOC_SHIM_IS_UNSTABLE, default_fn_name, global_fn_name,
};
use rustc_codegen_ssa::traits::{
    AbiBuilderMethods as _, BaseTypeCodegenMethods as _, BuilderMethods as _,
    ConstCodegenMethods as _,
};
use rustc_session::config::OomStrategy;
use rustc_span::DUMMY_SP;
use rustc_symbol_mangling::mangle_internal_symbol;

pub(crate) fn codegen(cx: &CodegenCx<'_>, methods: &[AllocatorMethod]) {
    let usize = cx.type_usize();
    let i8 = cx.type_i8();
    let i8p = cx.type_ptr_to(i8);

    for &method in methods {
        let mut method = method;

        // HACK(eddyb) https://github.com/rust-lang/rust/pull/150757
        // ("Fix `alloc_error_handler` signature mismatch") backport.
        if method.name == rustc_ast::expand::allocator::ALLOC_ERROR_HANDLER {
            assert_eq!(method.inputs.len(), 0);
            method.inputs = &[rustc_ast::expand::allocator::AllocatorMethodInput {
                name: "layout",
                ty: AllocatorTy::Layout,
            }];
        }

        let mut args = Vec::with_capacity(method.inputs.len());
        for input in method.inputs.iter() {
            match input.ty {
                AllocatorTy::Layout => {
                    args.push(usize); // size
                    args.push(usize); // align
                }
                AllocatorTy::Ptr => args.push(i8p),
                AllocatorTy::Usize => args.push(usize),

                AllocatorTy::Never | AllocatorTy::ResultPtr | AllocatorTy::Unit => {
                    panic!("invalid allocator arg")
                }
            }
        }

        let mut no_return = false;
        let output = match method.output {
            AllocatorTy::ResultPtr => Some(i8p),
            AllocatorTy::Unit => None,
            AllocatorTy::Never => {
                no_return = true;
                None
            }

            AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => {
                panic!("invalid allocator output")
            }
        };

        let from_name = mangle_internal_symbol(cx.tcx, &global_fn_name(method.name));
        let to_name = mangle_internal_symbol(cx.tcx, &default_fn_name(method.name));

        create_wrapper_function(cx, &from_name, Some(&to_name), &args, output, no_return);
    }

    // __rust_alloc_error_handler_should_panic_v2
    create_const_value_function(
        cx,
        &mangle_internal_symbol(cx.tcx, OomStrategy::SYMBOL),
        cx.const_i8(cx.tcx.sess.opts.unstable_opts.oom.should_panic() as i8),
    );

    // __rust_no_alloc_shim_is_unstable_v2
    create_wrapper_function(
        cx,
        &mangle_internal_symbol(cx.tcx, NO_ALLOC_SHIM_IS_UNSTABLE),
        None,
        &[],
        None,
        false,
    );
}

fn decl_fn(
    cx: &CodegenCx<'_>,
    name: &str,
    linkage_type: LinkageType,
    // NOTE(eddyb) these are SPIR-V type IDs.
    args: &[Word],
    output: Option<Word>,
    _no_return: bool,
) -> SpirvFunctionCursor {
    let ret_ty = output.unwrap_or_else(|| SpirvType::Void.def(DUMMY_SP, cx));
    let fn_ty = cx.type_func(args, ret_ty);

    let mut emit = cx.emit_global();
    let fn_id = emit
        .begin_function(ret_ty, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    for &ty in args {
        emit.function_parameter(ty).unwrap();
    }
    let index_in_builder = emit.selected_function().unwrap();
    emit.end_function().unwrap();
    drop(emit);
    cx.set_linkage(fn_id, name.to_string(), linkage_type);
    SpirvFunctionCursor {
        ty: fn_ty,
        id: fn_id,
        index_in_builder,
    }
}

fn create_const_value_function(cx: &CodegenCx<'_>, name: &str, value: SpirvValue) {
    let wrapper = decl_fn(cx, name, LinkageType::Export, &[], Some(value.ty), false);
    let mut bx = Builder::build(cx, Builder::append_block(cx, wrapper, ""));
    bx.ret(value);
}

fn create_wrapper_function(
    cx: &CodegenCx<'_>,
    from_name: &str,
    to_name: Option<&str>,
    // NOTE(eddyb) these are SPIR-V type IDs.
    args: &[Word],
    output: Option<Word>,
    no_return: bool,
) {
    let wrapper = decl_fn(cx, from_name, LinkageType::Export, args, output, no_return);
    let mut bx = Builder::build(cx, Builder::append_block(cx, wrapper, ""));

    if let Some(to_name) = to_name {
        let callee = decl_fn(cx, to_name, LinkageType::Import, args, output, no_return);
        let call_args = (0..args.len()).map(|i| bx.get_param(i)).collect::<Vec<_>>();
        let ret = bx.call(
            callee.ty,
            None,
            None,
            cx.def_constant(
                cx.type_ptr_to_ext(callee.ty, cx.data_layout().instruction_address_space),
                SpirvConst::PtrToFunc { func_id: callee.id },
            ),
            &call_args,
            None,
            None,
        );
        if output.is_some() {
            bx.ret(ret);
        } else {
            bx.ret_void();
        }
    } else {
        assert!(output.is_none());
        bx.ret_void();
    }
}
