pub use kajit_abi::*;

#[cfg(test)]
mod tests {
    use super::EncodeContext;
    use crate::intrinsics::kajit_output_grow;

    #[test]
    fn kajit_output_grow_intrinsic() {
        let mut ctx = EncodeContext::with_capacity(4);
        unsafe {
            *ctx.output_ptr = 0xAA;
            ctx.output_ptr = ctx.output_ptr.add(1);
            *ctx.output_ptr = 0xBB;
            ctx.output_ptr = ctx.output_ptr.add(1);
        }

        unsafe { kajit_output_grow(&mut ctx, 100) };
        assert_eq!(ctx.error.code, 0);
        assert_eq!(ctx.written(), 2);
        assert!(ctx.remaining() >= 100);

        let vec = ctx.into_vec();
        assert_eq!(&vec[..2], &[0xAA, 0xBB]);
    }
}
