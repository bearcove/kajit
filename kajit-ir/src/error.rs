use core::fmt;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    Ok = 0,
    UnexpectedEof = 1,
    InvalidVarint = 2,
    InvalidUtf8 = 3,
    UnsupportedShape = 4,
    ExpectedObjectStart = 5,
    ExpectedColon = 6,
    ExpectedStringKey = 7,
    UnterminatedString = 8,
    InvalidJsonNumber = 9,
    MissingRequiredField = 10,
    UnexpectedCharacter = 11,
    NumberOutOfRange = 12,
    InvalidBool = 13,
    UnknownVariant = 14,
    ExpectedTagKey = 15,
    AmbiguousVariant = 16,
    AllocError = 17,
    InvalidEscapeSequence = 18,
    UnknownField = 19,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCode::Ok => write!(f, "no error"),
            ErrorCode::UnexpectedEof => write!(f, "unexpected end of input"),
            ErrorCode::InvalidVarint => write!(f, "invalid varint encoding"),
            ErrorCode::InvalidUtf8 => write!(f, "invalid UTF-8"),
            ErrorCode::UnsupportedShape => write!(f, "unsupported shape"),
            ErrorCode::ExpectedObjectStart => write!(f, "expected '{{' to start object"),
            ErrorCode::ExpectedColon => write!(f, "expected ':' after key"),
            ErrorCode::ExpectedStringKey => write!(f, "expected '\"' to start key"),
            ErrorCode::UnterminatedString => write!(f, "unterminated string"),
            ErrorCode::InvalidJsonNumber => write!(f, "invalid JSON number"),
            ErrorCode::MissingRequiredField => write!(f, "missing required field"),
            ErrorCode::UnexpectedCharacter => write!(f, "unexpected character"),
            ErrorCode::NumberOutOfRange => write!(f, "number out of range for target type"),
            ErrorCode::InvalidBool => write!(f, "invalid bool value"),
            ErrorCode::UnknownVariant => write!(f, "unknown enum variant"),
            ErrorCode::ExpectedTagKey => write!(f, "expected tag key to appear first"),
            ErrorCode::AmbiguousVariant => write!(f, "ambiguous variant: multiple variants match"),
            ErrorCode::AllocError => write!(f, "memory allocation failed"),
            ErrorCode::InvalidEscapeSequence => write!(f, "invalid JSON escape sequence"),
            ErrorCode::UnknownField => write!(f, "unknown field"),
        }
    }
}
