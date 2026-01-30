//! C type aliases for FFI.
//!
//! These types map to their C equivalents and are used in extern blocks
//! for declaring foreign functions.
//!
//! # Example
//! ```sigil
//! use ffi::c::*;
//!
//! extern "C" {
//!     fn strlen(s: *const c_char) -> usize;
//!     fn malloc(size: usize) -> *mut c_void;
//!     fn free(ptr: *mut c_void);
//!     fn printf(fmt: *const c_char, ...) -> c_int;
//! }
//! ```

/// C type information for type checking and code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CType {
    // Void type
    Void,

    // Character types
    Char,  // c_char (signed or unsigned depending on platform)
    SChar, // c_schar (signed char)
    UChar, // c_uchar (unsigned char)

    // Integer types
    Short,     // c_short
    UShort,    // c_ushort
    Int,       // c_int
    UInt,      // c_uint
    Long,      // c_long
    ULong,     // c_ulong
    LongLong,  // c_longlong
    ULongLong, // c_ulonglong

    // Floating point
    Float,  // c_float
    Double, // c_double

    // Size types
    Size,    // size_t (usize in Sigil)
    SSize,   // ssize_t (isize in Sigil)
    PtrDiff, // ptrdiff_t

    // Fixed-width integers (from stdint.h)
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl CType {
    /// Get the Sigil type name for this C type.
    pub fn sigil_name(&self) -> &'static str {
        match self {
            CType::Void => "c_void",
            CType::Char => "c_char",
            CType::SChar => "c_schar",
            CType::UChar => "c_uchar",
            CType::Short => "c_short",
            CType::UShort => "c_ushort",
            CType::Int => "c_int",
            CType::UInt => "c_uint",
            CType::Long => "c_long",
            CType::ULong => "c_ulong",
            CType::LongLong => "c_longlong",
            CType::ULongLong => "c_ulonglong",
            CType::Float => "c_float",
            CType::Double => "c_double",
            CType::Size => "size_t",
            CType::SSize => "ssize_t",
            CType::PtrDiff => "ptrdiff_t",
            CType::Int8 => "i8",
            CType::Int16 => "i16",
            CType::Int32 => "i32",
            CType::Int64 => "i64",
            CType::UInt8 => "u8",
            CType::UInt16 => "u16",
            CType::UInt32 => "u32",
            CType::UInt64 => "u64",
        }
    }

    /// Get the size in bytes on the target platform (assuming LP64).
    pub fn size_bytes(&self) -> usize {
        match self {
            CType::Void => 0,
            CType::Char | CType::SChar | CType::UChar => 1,
            CType::Short | CType::UShort => 2,
            CType::Int | CType::UInt => 4,
            CType::Long | CType::ULong => 8, // LP64
            CType::LongLong | CType::ULongLong => 8,
            CType::Float => 4,
            CType::Double => 8,
            CType::Size | CType::SSize | CType::PtrDiff => 8, // 64-bit
            CType::Int8 | CType::UInt8 => 1,
            CType::Int16 | CType::UInt16 => 2,
            CType::Int32 | CType::UInt32 => 4,
            CType::Int64 | CType::UInt64 => 8,
        }
    }

    /// Check if this is a signed integer type.
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            CType::SChar
                | CType::Short
                | CType::Int
                | CType::Long
                | CType::LongLong
                | CType::SSize
                | CType::PtrDiff
                | CType::Int8
                | CType::Int16
                | CType::Int32
                | CType::Int64
        )
    }

    /// Parse a C type name to its enum variant.
    pub fn from_name(name: &str) -> Option<CType> {
        match name {
            "c_void" | "void" => Some(CType::Void),
            "c_char" | "char" => Some(CType::Char),
            "c_schar" => Some(CType::SChar),
            "c_uchar" => Some(CType::UChar),
            "c_short" | "short" => Some(CType::Short),
            "c_ushort" => Some(CType::UShort),
            "c_int" | "int" => Some(CType::Int),
            "c_uint" => Some(CType::UInt),
            "c_long" | "long" => Some(CType::Long),
            "c_ulong" => Some(CType::ULong),
            "c_longlong" => Some(CType::LongLong),
            "c_ulonglong" => Some(CType::ULongLong),
            "c_float" | "float" => Some(CType::Float),
            "c_double" | "double" => Some(CType::Double),
            "size_t" | "usize" => Some(CType::Size),
            "ssize_t" | "isize" => Some(CType::SSize),
            "ptrdiff_t" => Some(CType::PtrDiff),
            "i8" | "int8_t" => Some(CType::Int8),
            "i16" | "int16_t" => Some(CType::Int16),
            "i32" | "int32_t" => Some(CType::Int32),
            "i64" | "int64_t" => Some(CType::Int64),
            "u8" | "uint8_t" => Some(CType::UInt8),
            "u16" | "uint16_t" => Some(CType::UInt16),
            "u32" | "uint32_t" => Some(CType::UInt32),
            "u64" | "uint64_t" => Some(CType::UInt64),
            _ => None,
        }
    }
}

/// C type aliases as they appear in Sigil code.
pub const C_TYPE_ALIASES: &[(&str, &str)] = &[
    // Void
    ("c_void", "()"),
    // Character types
    ("c_char", "i8"), // Platform-dependent, assume signed
    ("c_schar", "i8"),
    ("c_uchar", "u8"),
    // Integer types (LP64 model)
    ("c_short", "i16"),
    ("c_ushort", "u16"),
    ("c_int", "i32"),
    ("c_uint", "u32"),
    ("c_long", "i64"), // LP64: long is 64-bit
    ("c_ulong", "u64"),
    ("c_longlong", "i64"),
    ("c_ulonglong", "u64"),
    // Floating point
    ("c_float", "f32"),
    ("c_double", "f64"),
    // Size types
    ("size_t", "usize"),
    ("ssize_t", "isize"),
    ("ptrdiff_t", "isize"),
    // Fixed-width (these are just aliases)
    ("int8_t", "i8"),
    ("int16_t", "i16"),
    ("int32_t", "i32"),
    ("int64_t", "i64"),
    ("uint8_t", "u8"),
    ("uint16_t", "u16"),
    ("uint32_t", "u32"),
    ("uint64_t", "u64"),
];

/// Check if a type name is a C type.
pub fn is_c_type(name: &str) -> bool {
    CType::from_name(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_type_parsing() {
        assert_eq!(CType::from_name("c_int"), Some(CType::Int));
        assert_eq!(CType::from_name("c_void"), Some(CType::Void));
        assert_eq!(CType::from_name("size_t"), Some(CType::Size));
        assert_eq!(CType::from_name("unknown"), None);
    }

    #[test]
    fn test_c_type_sizes() {
        assert_eq!(CType::Int.size_bytes(), 4);
        assert_eq!(CType::Long.size_bytes(), 8);
        assert_eq!(CType::Char.size_bytes(), 1);
        assert_eq!(CType::Double.size_bytes(), 8);
    }

    #[test]
    fn test_c_type_signedness() {
        assert!(CType::Int.is_signed());
        assert!(!CType::UInt.is_signed());
        assert!(CType::SSize.is_signed());
        assert!(!CType::Size.is_signed());
    }
}
