use super::CodegenCx;
use crate::abi::ConvSpirvType;
use crate::spirv_type::SpirvType;
use rspirv::spirv::Word;
use rustc_codegen_ssa::common::TypeKind;
use rustc_codegen_ssa::traits::{
    BaseTypeMethods, ConstMethods as _, DerivedTypeMethods as _, LayoutTypeMethods,
};
use rustc_middle::ty::layout::{
    FnAbiError, FnAbiOfHelpers, FnAbiRequest, LayoutError, LayoutOfHelpers, TyAndLayout,
};
use rustc_middle::ty::Ty;
use rustc_middle::{bug, span_bug};
use rustc_span::source_map::Spanned;
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::call::{CastTarget, FnAbi, Reg};
use rustc_target::abi::{Abi, AddressSpace, Align, Integer, Size};

impl<'tcx> LayoutOfHelpers<'tcx> for CodegenCx<'tcx> {
    type LayoutOfResult = TyAndLayout<'tcx>;

    #[inline]
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        if let LayoutError::SizeOverflow(_) = err {
            self.tcx.dcx().span_fatal(span, err.to_string())
        } else {
            span_bug!(span, "failed to get layout for `{}`: {}", ty, err)
        }
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for CodegenCx<'tcx> {
    type FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>;

    #[inline]
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        if let FnAbiError::Layout(LayoutError::SizeOverflow(_)) = err {
            self.tcx.dcx().emit_fatal(Spanned { span, node: err })
        } else {
            match fn_abi_request {
                FnAbiRequest::OfFnPtr { sig, extra_args } => {
                    span_bug!(
                        span,
                        "`fn_abi_of_fn_ptr({sig}, {extra_args:?})` failed: {err:?}",
                    );
                }
                FnAbiRequest::OfInstance {
                    instance,
                    extra_args,
                } => {
                    span_bug!(
                        span,
                        "`fn_abi_of_instance({instance}, {extra_args:?})` failed: {err:?}",
                    );
                }
            }
        }
    }
}

impl<'tcx> LayoutTypeMethods<'tcx> for CodegenCx<'tcx> {
    fn backend_type(&self, layout: TyAndLayout<'tcx>) -> Self::Type {
        layout.spirv_type(DUMMY_SP, self)
    }

    fn cast_backend_type(&self, ty: &CastTarget) -> Self::Type {
        bug!(
            "cast_backend_type({:?}): query hooks should've made `PassMode::Cast` impossible",
            ty
        )
    }

    fn fn_decl_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> Self::Type {
        fn_abi.spirv_type(DUMMY_SP, self)
    }

    fn fn_ptr_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> Self::Type {
        self.type_ptr_to(self.fn_decl_backend_type(fn_abi))
    }

    fn reg_backend_type(&self, ty: &Reg) -> Self::Type {
        bug!(
            "reg_backend_type({:?}): query hooks should've made `PassMode::Cast` impossible",
            ty
        )
    }

    fn immediate_backend_type(&self, layout: TyAndLayout<'tcx>) -> Self::Type {
        layout.spirv_type(DUMMY_SP, self)
    }

    fn is_backend_immediate(&self, layout: TyAndLayout<'tcx>) -> bool {
        match layout.abi {
            Abi::Scalar(_) | Abi::Vector { .. } => true,
            Abi::ScalarPair(..) => false,
            Abi::Uninhabited | Abi::Aggregate { .. } => layout.is_zst(),
        }
    }

    fn is_backend_scalar_pair(&self, layout: TyAndLayout<'tcx>) -> bool {
        match layout.abi {
            Abi::ScalarPair(..) => true,
            Abi::Uninhabited | Abi::Scalar(_) | Abi::Vector { .. } | Abi::Aggregate { .. } => false,
        }
    }

    fn scalar_pair_element_backend_type(
        &self,
        layout: TyAndLayout<'tcx>,
        index: usize,
        _immediate: bool,
    ) -> Self::Type {
        crate::abi::scalar_pair_element_backend_type(self, DUMMY_SP, layout, index)
    }
}

impl<'tcx> CodegenCx<'tcx> {
    pub fn type_usize(&self) -> Word {
        let ptr_size = self.tcx.data_layout.pointer_size.bits() as u32;
        SpirvType::Integer(ptr_size, false).def(DUMMY_SP, self)
    }

    /// Find the largest integer with the given alignment or less, preferring
    /// supported types over unsupported ones (only used as a last resort).
    //
    // HACK(eddyb) this is a copy of `Integer::approximate_align` that also
    // checks for SPIR-V capabilities being enabled.
    pub fn integer_approximate_align_prefer_legal(&self, wanted: Align) -> Integer {
        use Integer::*;

        // HACK(eddyb) this loop shouldn't really exist like this - instead,
        // which integer types are supported should be kept in `ContextCx`
        // (e.g. pair of `ArrayVec<[Integer; 5]>`s, or even an iterable bitset),
        // or even a `log2(align)`-indexed precomputed `int_approx_align` array
        // (w/o storing the tail of equal results for increasing alignments).
        let mut illegal_fallback = None;
        for candidate in [I64, I32, I16] {
            if candidate.align(self).abi > wanted || candidate.size().bytes() > wanted.bytes() {
                continue;
            }

            let legal_candidate = crate::spirv_type::integer_type_capability(candidate)
                .map_or(true, |cap| self.builder.has_capability(cap));
            if legal_candidate {
                return candidate;
            }

            // HACK(eddyb) the loop only keeps going in case a smaller integer
            // is actually supported.
            if illegal_fallback.is_none() {
                illegal_fallback = Some(candidate);
            }
        }
        illegal_fallback.unwrap_or(I8)
    }

    /// Return a SPIR-V type that has at most the required alignment,
    /// and exactly the required size, as a best-effort padding array.
    pub(crate) fn type_padding_filler(&self, size: Size, align: Align) -> Word {
        let unit = self.integer_approximate_align_prefer_legal(align);
        let size = size.bytes();
        let unit_size = unit.size().bytes();
        assert_eq!(size % unit_size, 0);
        self.type_array(self.type_from_integer(unit), size / unit_size)
    }
}

impl<'tcx> BaseTypeMethods<'tcx> for CodegenCx<'tcx> {
    fn type_i1(&self) -> Self::Type {
        SpirvType::Bool.def(DUMMY_SP, self)
    }
    fn type_i8(&self) -> Self::Type {
        SpirvType::Integer(8, false).def(DUMMY_SP, self)
    }
    fn type_i16(&self) -> Self::Type {
        SpirvType::Integer(16, false).def(DUMMY_SP, self)
    }
    fn type_i32(&self) -> Self::Type {
        SpirvType::Integer(32, false).def(DUMMY_SP, self)
    }
    fn type_i64(&self) -> Self::Type {
        SpirvType::Integer(64, false).def(DUMMY_SP, self)
    }
    fn type_i128(&self) -> Self::Type {
        SpirvType::Integer(128, false).def(DUMMY_SP, self)
    }
    fn type_isize(&self) -> Self::Type {
        let ptr_size = self.tcx.data_layout.pointer_size.bits() as u32;
        SpirvType::Integer(ptr_size, false).def(DUMMY_SP, self)
    }

    fn type_f16(&self) -> Self::Type {
        SpirvType::Float(16).def(DUMMY_SP, self)
    }
    fn type_f32(&self) -> Self::Type {
        SpirvType::Float(32).def(DUMMY_SP, self)
    }
    fn type_f64(&self) -> Self::Type {
        SpirvType::Float(64).def(DUMMY_SP, self)
    }
    fn type_f128(&self) -> Self::Type {
        SpirvType::Float(128).def(DUMMY_SP, self)
    }

    fn type_array(&self, ty: Self::Type, len: u64) -> Self::Type {
        SpirvType::Array {
            element: ty,
            count: self.const_usize(len),
        }
        .def(DUMMY_SP, self)
    }

    fn type_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type {
        SpirvType::Function {
            return_type: ret,
            arguments: args,
        }
        .def(DUMMY_SP, self)
    }
    fn type_struct(&self, els: &[Self::Type], _packed: bool) -> Self::Type {
        // FIXME(eddyb) use `AccumulateVec`s just like `rustc` itself does.
        let (field_offsets, size, align) = crate::abi::auto_struct_layout(self, els);
        SpirvType::Adt {
            def_id: None,
            align,
            size,
            field_types: els,
            field_offsets: &field_offsets,
            field_names: None,
        }
        .def(DUMMY_SP, self)
    }
    fn type_kind(&self, ty: Self::Type) -> TypeKind {
        match self.lookup_type(ty) {
            SpirvType::Void => TypeKind::Void,
            SpirvType::Bool | // thanks llvm
            SpirvType::Integer(_, _) => TypeKind::Integer,
            SpirvType::Float(width) => match width {
                16 => TypeKind::Half,
                32 => TypeKind::Float,
                64 => TypeKind::Double,
                128 => TypeKind::FP128,
                other => self
                    .tcx
                    .dcx()
                    .fatal(format!("Invalid float width in type_kind: {other}")),
            },
            SpirvType::Adt { .. } | SpirvType::InterfaceBlock { .. } => {
                TypeKind::Struct
            }
            SpirvType::Vector { .. } => TypeKind::Vector,
            SpirvType::Array { .. } | SpirvType::RuntimeArray { .. } | SpirvType::Matrix { .. } => TypeKind::Array,
            SpirvType::Pointer { .. } => TypeKind::Pointer,
            SpirvType::Function { .. } => TypeKind::Function,
            // HACK(eddyb) this is probably the closest `TypeKind` (which is still
            // very much LLVM-specific, sadly) has to offer to "resource handle".
            | SpirvType::Image { .. }
            | SpirvType::Sampler
            | SpirvType::SampledImage { .. }
            | SpirvType::AccelerationStructureKhr
            | SpirvType::RayQueryKhr
                => TypeKind::Token,
        }
    }
    fn type_ptr(&self) -> Self::Type {
        self.type_ptr_to(SpirvType::Void.def(DUMMY_SP, self))
    }
    fn type_ptr_ext(&self, address_space: AddressSpace) -> Self::Type {
        self.type_ptr_to_ext(SpirvType::Void.def(DUMMY_SP, self), address_space)
    }
    fn element_type(&self, ty: Self::Type) -> Self::Type {
        match self.lookup_type(ty) {
            SpirvType::Pointer { pointee } => pointee,
            SpirvType::Vector { element, .. } => element,
            spirv_type => self.tcx.dcx().fatal(format!(
                "element_type called on invalid type: {spirv_type:?}"
            )),
        }
    }

    /// Returns the number of elements in `self` if it is a LLVM vector type.
    fn vector_length(&self, ty: Self::Type) -> usize {
        match self.lookup_type(ty) {
            SpirvType::Vector { count, .. } => count as usize,
            ty => self
                .tcx
                .dcx()
                .fatal(format!("vector_length called on non-vector type: {ty:?}")),
        }
    }

    fn float_width(&self, ty: Self::Type) -> usize {
        match self.lookup_type(ty) {
            SpirvType::Float(width) => width as usize,
            ty => self
                .tcx
                .dcx()
                .fatal(format!("float_width called on non-float type: {ty:?}")),
        }
    }

    /// Retrieves the bit width of the integer type `self`.
    fn int_width(&self, ty: Self::Type) -> u64 {
        match self.lookup_type(ty) {
            SpirvType::Integer(width, _) => width as u64,
            ty => self
                .tcx
                .dcx()
                .fatal(format!("int_width called on non-integer type: {ty:?}")),
        }
    }

    fn val_ty(&self, v: Self::Value) -> Self::Type {
        v.ty
    }
}
