use kajit_abi::DeserContext;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntrinsicCallError {
    UnsupportedArity { kind: &'static str, arity: usize },
}

impl core::fmt::Display for IntrinsicCallError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedArity { kind, arity } => {
                write!(f, "{kind} intrinsic call has unsupported arity: {arity}")
            }
        }
    }
}

impl std::error::Error for IntrinsicCallError {}

pub unsafe fn call_pure_u64(func: usize, args: &[u64]) -> Result<u64, IntrinsicCallError> {
    macro_rules! pure_fn {
        () => {
            unsafe extern "C" fn() -> u64
        };
        ($($arg:ty),+) => {
            unsafe extern "C" fn($($arg),+) -> u64
        };
    }

    Ok(match args.len() {
        0 => unsafe { (core::mem::transmute::<usize, pure_fn!()>(func))() },
        1 => unsafe { (core::mem::transmute::<usize, pure_fn!(u64)>(func))(args[0]) },
        2 => unsafe { (core::mem::transmute::<usize, pure_fn!(u64, u64)>(func))(args[0], args[1]) },
        3 => unsafe {
            (core::mem::transmute::<usize, pure_fn!(u64, u64, u64)>(func))(
                args[0], args[1], args[2],
            )
        },
        4 => unsafe {
            (core::mem::transmute::<usize, pure_fn!(u64, u64, u64, u64)>(func))(
                args[0], args[1], args[2], args[3],
            )
        },
        5 => unsafe {
            (core::mem::transmute::<usize, pure_fn!(u64, u64, u64, u64, u64)>(func))(
                args[0], args[1], args[2], args[3], args[4],
            )
        },
        6 => unsafe {
            (core::mem::transmute::<usize, pure_fn!(u64, u64, u64, u64, u64, u64)>(func))(
                args[0], args[1], args[2], args[3], args[4], args[5],
            )
        },
        7 => unsafe {
            (core::mem::transmute::<usize, pure_fn!(u64, u64, u64, u64, u64, u64, u64)>(func))(
                args[0], args[1], args[2], args[3], args[4], args[5], args[6],
            )
        },
        8 => unsafe {
            (core::mem::transmute::<usize, pure_fn!(u64, u64, u64, u64, u64, u64, u64, u64)>(func))(
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
            )
        },
        arity => {
            return Err(IntrinsicCallError::UnsupportedArity {
                kind: "CallPure",
                arity,
            });
        }
    })
}

pub unsafe fn call_intrinsic_with_result(
    func: usize,
    ctx: *mut DeserContext,
    args: &[u64],
) -> Result<u64, IntrinsicCallError> {
    macro_rules! intrinsic_fn {
        () => {
            unsafe extern "C" fn(*mut DeserContext) -> u64
        };
        ($($arg:ty),+) => {
            unsafe extern "C" fn(*mut DeserContext, $($arg),+) -> u64
        };
    }

    Ok(match args.len() {
        0 => unsafe { (core::mem::transmute::<usize, intrinsic_fn!()>(func))(ctx) },
        1 => unsafe { (core::mem::transmute::<usize, intrinsic_fn!(u64)>(func))(ctx, args[0]) },
        2 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn!(u64, u64)>(func))(ctx, args[0], args[1])
        },
        3 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn!(u64, u64, u64)>(func))(
                ctx, args[0], args[1], args[2],
            )
        },
        4 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn!(u64, u64, u64, u64)>(func))(
                ctx, args[0], args[1], args[2], args[3],
            )
        },
        5 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn!(u64, u64, u64, u64, u64)>(func))(
                ctx, args[0], args[1], args[2], args[3], args[4],
            )
        },
        6 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn!(u64, u64, u64, u64, u64, u64)>(func))(
                ctx, args[0], args[1], args[2], args[3], args[4], args[5],
            )
        },
        7 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn!(u64, u64, u64, u64, u64, u64, u64)>(func))(
                ctx, args[0], args[1], args[2], args[3], args[4], args[5], args[6],
            )
        },
        8 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn!(u64, u64, u64, u64, u64, u64, u64, u64)>(
                func,
            ))(
                ctx, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
            )
        },
        arity => {
            return Err(IntrinsicCallError::UnsupportedArity {
                kind: "CallIntrinsic(result)",
                arity,
            });
        }
    })
}

pub unsafe fn call_intrinsic_with_output(
    func: usize,
    ctx: *mut DeserContext,
    args: &[u64],
    out_ptr: *mut u8,
) -> Result<(), IntrinsicCallError> {
    macro_rules! intrinsic_fn_void {
        () => {
            unsafe extern "C" fn(*mut DeserContext, *mut u8)
        };
        ($($arg:ty),+) => {
            unsafe extern "C" fn(*mut DeserContext, $($arg),+, *mut u8)
        };
    }

    match args.len() {
        0 => unsafe { (core::mem::transmute::<usize, intrinsic_fn_void!()>(func))(ctx, out_ptr) },
        1 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn_void!(u64)>(func))(ctx, args[0], out_ptr)
        },
        2 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn_void!(u64, u64)>(func))(
                ctx, args[0], args[1], out_ptr,
            )
        },
        3 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn_void!(u64, u64, u64)>(func))(
                ctx, args[0], args[1], args[2], out_ptr,
            )
        },
        4 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn_void!(u64, u64, u64, u64)>(func))(
                ctx, args[0], args[1], args[2], args[3], out_ptr,
            )
        },
        5 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn_void!(u64, u64, u64, u64, u64)>(func))(
                ctx, args[0], args[1], args[2], args[3], args[4], out_ptr,
            )
        },
        6 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn_void!(u64, u64, u64, u64, u64, u64)>(func))(
                ctx, args[0], args[1], args[2], args[3], args[4], args[5], out_ptr,
            )
        },
        7 => unsafe {
            (core::mem::transmute::<usize, intrinsic_fn_void!(u64, u64, u64, u64, u64, u64, u64)>(
                func,
            ))(
                ctx, args[0], args[1], args[2], args[3], args[4], args[5], args[6], out_ptr,
            )
        },
        8 => unsafe {
            (core::mem::transmute::<
                usize,
                intrinsic_fn_void!(u64, u64, u64, u64, u64, u64, u64, u64),
            >(func))(
                ctx, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
                out_ptr,
            )
        },
        arity => {
            return Err(IntrinsicCallError::UnsupportedArity {
                kind: "CallIntrinsic(output)",
                arity,
            });
        }
    };
    Ok(())
}
