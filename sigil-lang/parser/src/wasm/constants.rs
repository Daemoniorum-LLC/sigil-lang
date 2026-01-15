//! WASM compilation constants.
//!
//! Defines evidence tags, type tags, and memory layout for the WASM runtime.

/// Evidentiality tag bits (stored in high bits of i64).
///
/// Sigil's evidentiality system tracks data provenance:
/// - `!` (Known): Locally computed/verified
/// - `?` (Uncertain): May be absent or unknown
/// - `~` (Reported): External/untrusted source
/// - `‽` (Paradox): Trust boundary crossing
pub mod evidence {
    /// Known evidence marker (!)
    pub const KNOWN: i64 = 0x0000_0000_0000_0000;
    /// Uncertain evidence marker (?)
    pub const UNCERTAIN: i64 = 0x1000_0000_0000_0000;
    /// Reported evidence marker (~)
    pub const REPORTED: i64 = 0x2000_0000_0000_0000;
    /// Paradox evidence marker (‽)
    pub const PARADOX: i64 = 0x3000_0000_0000_0000;

    /// Mask for extracting evidence tag (top 4 bits)
    pub const TAG_MASK: i64 = 0x7000_0000_0000_0000;
    /// Mask for extracting value (without tag)
    pub const VALUE_MASK: i64 = 0x0FFF_FFFF_FFFF_FFFF;

    /// Bit shift for type tags
    pub const TYPE_SHIFT: u32 = 56;
}

/// Type tags (stored in bits 56-59).
///
/// Used for runtime type checking in WASM.
pub mod type_tag {
    /// Integer type
    pub const INT: i64 = 0x00 << 56;
    /// Float type
    pub const FLOAT: i64 = 0x01 << 56;
    /// Boolean type
    pub const BOOL: i64 = 0x02 << 56;
    /// Null/unit type
    pub const NULL: i64 = 0x03 << 56;
    /// Pointer type
    pub const PTR: i64 = 0x04 << 56;
    /// Function reference type
    pub const FUNC: i64 = 0x05 << 56;
    /// String type
    pub const STRING: i64 = 0x06 << 56;
    /// Array type
    pub const ARRAY: i64 = 0x07 << 56;
    /// Struct type
    pub const STRUCT: i64 = 0x08 << 56;
    /// Closure type
    pub const CLOSURE: i64 = 0x09 << 56;
}

/// Memory layout constants.
///
/// WASM linear memory organization:
/// ```text
/// 0x0000 - 0x03FF: Reserved (null pointer trap zone)
/// 0x0400 - 0x0FFF: Stack (3KB)
/// 0x1000 - 0x1FFF: Globals
/// 0x2000 - 0x2FFF: String pool
/// 0x3000 - 0x3FFF: VDOM pool
/// 0x4000+:         Heap (bump allocator)
/// ```
pub mod memory {
    /// Start of stack region
    pub const STACK_START: u32 = 0x0400;
    /// Size of stack (3KB)
    pub const STACK_SIZE: u32 = 0x0C00;
    /// Start of globals region
    pub const GLOBALS_START: u32 = 0x1000;
    /// Start of string pool
    pub const STRING_POOL_START: u32 = 0x2000;
    /// Start of VDOM pool
    pub const VDOM_POOL_START: u32 = 0x3000;
    /// Start of heap
    pub const HEAP_START: u32 = 0x4000;

    /// Initial memory pages (64KB each)
    pub const INITIAL_PAGES: u64 = 16;
    /// Maximum memory pages
    pub const MAX_PAGES: u64 = 256;

    /// Alignment for allocations (8 bytes)
    pub const ALIGNMENT: u32 = 8;
}

/// WASM instruction limits and configuration.
pub mod limits {
    /// Maximum function locals
    pub const MAX_LOCALS: u32 = 1024;
    /// Maximum nested blocks
    pub const MAX_BLOCK_DEPTH: u32 = 64;
    /// Maximum table elements
    pub const MAX_TABLE_SIZE: u32 = 4096;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evidence_tags_are_distinct() {
        assert_ne!(evidence::KNOWN, evidence::UNCERTAIN);
        assert_ne!(evidence::UNCERTAIN, evidence::REPORTED);
        assert_ne!(evidence::REPORTED, evidence::PARADOX);
    }

    #[test]
    fn test_evidence_mask_extracts_tag() {
        let tagged_value = 42i64 | evidence::REPORTED;
        let tag = tagged_value & evidence::TAG_MASK;
        assert_eq!(tag, evidence::REPORTED);
    }

    #[test]
    fn test_value_mask_extracts_value() {
        let original = 42i64;
        let tagged = original | evidence::UNCERTAIN;
        let extracted = tagged & evidence::VALUE_MASK;
        assert_eq!(extracted, original);
    }

    #[test]
    fn test_type_tags_fit_in_range() {
        // All type tags should be in bits 56-59
        let tags = [
            type_tag::INT,
            type_tag::FLOAT,
            type_tag::BOOL,
            type_tag::NULL,
            type_tag::PTR,
            type_tag::FUNC,
            type_tag::STRING,
            type_tag::ARRAY,
            type_tag::STRUCT,
            type_tag::CLOSURE,
        ];

        for tag in tags {
            // Should not overlap with evidence tags
            assert_eq!(tag & evidence::TAG_MASK, 0);
            // Should be in the type tag range
            assert!(tag >= (0x00i64 << 56));
            assert!(tag <= (0x0Fi64 << 56));
        }
    }

    #[test]
    fn test_type_tags_are_distinct() {
        let tags = [
            type_tag::INT,
            type_tag::FLOAT,
            type_tag::BOOL,
            type_tag::NULL,
            type_tag::PTR,
            type_tag::FUNC,
            type_tag::STRING,
            type_tag::ARRAY,
            type_tag::STRUCT,
            type_tag::CLOSURE,
        ];

        // Each tag should be unique
        for i in 0..tags.len() {
            for j in (i + 1)..tags.len() {
                assert_ne!(tags[i], tags[j], "Type tags at {} and {} should be distinct", i, j);
            }
        }
    }

    #[test]
    fn test_memory_layout_non_overlapping() {
        assert!(memory::STACK_START + memory::STACK_SIZE <= memory::GLOBALS_START);
        assert!(memory::GLOBALS_START < memory::STRING_POOL_START);
        assert!(memory::STRING_POOL_START < memory::VDOM_POOL_START);
        assert!(memory::VDOM_POOL_START < memory::HEAP_START);
    }

    #[test]
    fn test_heap_start_alignment() {
        assert_eq!(memory::HEAP_START % memory::ALIGNMENT, 0);
    }

    #[test]
    fn test_limits_constants() {
        // Limits should be reasonable values
        assert!(limits::MAX_LOCALS > 0);
        assert!(limits::MAX_BLOCK_DEPTH > 0);
        assert!(limits::MAX_TABLE_SIZE > 0);

        // Specific expected values
        assert_eq!(limits::MAX_LOCALS, 1024);
        assert_eq!(limits::MAX_BLOCK_DEPTH, 64);
        assert_eq!(limits::MAX_TABLE_SIZE, 4096);
    }

    #[test]
    fn test_evidence_and_type_tags_compatible() {
        // Evidence and type tags should be able to combine without overlap
        let value = 42i64;
        let typed_value = value | type_tag::INT;
        let evidenced_value = typed_value | evidence::REPORTED;

        // Should be able to extract both
        let evidence_tag = evidenced_value & evidence::TAG_MASK;
        assert_eq!(evidence_tag, evidence::REPORTED);

        // Type tag uses bits 56-59, evidence uses 60-62
        // The value bits and type tag should be preserved
        let value_and_type = evidenced_value & !evidence::TAG_MASK;
        assert_eq!(value_and_type, typed_value);
    }

    #[test]
    fn test_type_shift_constant() {
        // Type shift should be 56 bits
        assert_eq!(evidence::TYPE_SHIFT, 56);

        // Verify shift produces correct tag values
        let int_via_shift = 0x00i64 << evidence::TYPE_SHIFT;
        assert_eq!(int_via_shift, type_tag::INT);

        let float_via_shift = 0x01i64 << evidence::TYPE_SHIFT;
        assert_eq!(float_via_shift, type_tag::FLOAT);
    }

    #[test]
    fn test_memory_pages_config() {
        // Initial pages should be less than max
        assert!(memory::INITIAL_PAGES < memory::MAX_PAGES);

        // 16 initial pages = 1MB, 256 max pages = 16MB
        assert_eq!(memory::INITIAL_PAGES, 16);
        assert_eq!(memory::MAX_PAGES, 256);
    }

    #[test]
    fn test_stack_size_sufficient() {
        // Stack should be at least 3KB
        assert!(memory::STACK_SIZE >= 0x0C00);
    }

    #[test]
    fn test_all_regions_aligned() {
        // All region starts should be 8-byte aligned
        assert_eq!(memory::STACK_START % memory::ALIGNMENT, 0);
        assert_eq!(memory::GLOBALS_START % memory::ALIGNMENT, 0);
        assert_eq!(memory::STRING_POOL_START % memory::ALIGNMENT, 0);
        assert_eq!(memory::VDOM_POOL_START % memory::ALIGNMENT, 0);
        assert_eq!(memory::HEAP_START % memory::ALIGNMENT, 0);
    }
}
