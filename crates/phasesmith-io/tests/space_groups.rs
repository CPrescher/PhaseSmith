//! Native space-group database contract tests.

use phasesmith_crystallography::CrystalSystem;
use phasesmith_io::{
    SPACE_GROUP_DATABASE_PROVENANCE, SpaceGroupLookupError, space_group_by_hall_symbol,
    space_group_by_number, space_group_by_symbol, space_group_from_hall_symbol,
};

#[test]
fn number_lookup_returns_exact_conventional_operations() {
    let info = space_group_by_number(225).expect("F m -3 m must resolve");
    assert_eq!(info.number, 225);
    assert_eq!(info.hm_symbol, "F m -3 m");
    assert!(!info.hall_symbol.is_empty());
    assert_eq!(info.space_group.operations().len(), 192);
    assert!(
        info.space_group
            .is_systematically_absent([1, 0, 0])
            .unwrap()
    );
    assert!(
        !info
            .space_group
            .is_systematically_absent([1, 1, 1])
            .unwrap()
    );
    assert!(
        !info
            .space_group
            .is_systematically_absent([2, 0, 0])
            .unwrap()
    );
}

#[test]
fn symbols_preserve_explicit_settings_and_choose_standard_short_setting() {
    let short = space_group_by_symbol("P 21/c").expect("short symbol must resolve");
    assert_eq!(short.number, 14);
    assert_eq!(short.setting, "b1");
    assert_eq!(short.space_group.operations().len(), 4);

    let full = space_group_by_symbol("P 1 21/c 1").expect("full symbol must resolve");
    assert_eq!(full.hall_symbol, short.hall_symbol);
    assert_eq!(full.space_group, short.space_group);

    let rhombohedral = space_group_by_symbol("R -3 c :H").expect("H qualifier must resolve");
    assert_eq!(rhombohedral.number, 167);
    assert_eq!(rhombohedral.setting, "H");
    assert_eq!(rhombohedral.space_group.operations().len(), 36);

    let rhombohedral_axes = space_group_by_symbol("R -3 c :R").expect("R qualifier must resolve");
    assert_eq!(
        rhombohedral_axes.space_group.crystal_system(),
        CrystalSystem::Trigonal
    );
}

#[test]
fn hall_symbols_and_invalid_inputs_are_explicit() {
    let hall = space_group_by_symbol("-F 4 2 3").expect("Hall symbol must resolve");
    assert_eq!(hall.number, 225);
    assert_eq!(
        hall.space_group,
        space_group_by_number(225).unwrap().space_group
    );
    let cif_hall = space_group_by_hall_symbol("-R 3 2\"c").unwrap();
    assert_eq!(cif_hall.number, 167);
    assert_eq!(
        cif_hall.hall_symbol, "-R 3 2\"c",
        "the public symbol must use conventional CIF Hall quote syntax"
    );

    assert_eq!(
        space_group_by_number(0),
        Err(SpaceGroupLookupError::InvalidNumber)
    );
    assert!(matches!(
        space_group_by_symbol("not a space group"),
        Err(SpaceGroupLookupError::UnknownSymbol { .. })
    ));
}

#[test]
fn general_hall_parser_accepts_equivalent_noncanonical_expressions() {
    let canonical = space_group_from_hall_symbol("A 2 -2ab").unwrap();
    let redundant = space_group_from_hall_symbol("A 2 -2ac").unwrap();
    assert_eq!(redundant, canonical);
    assert_eq!(redundant.operations().len(), 8);
    assert_eq!(
        redundant,
        space_group_by_number(41).unwrap().space_group,
        "the redundant SHELX-era Hall spelling must retain exact No. 41 symmetry"
    );

    let shifted = space_group_from_hall_symbol("P 2y (3 0 0)").unwrap();
    assert_eq!(shifted.operations().len(), 2);
    assert_ne!(
        shifted,
        space_group_from_hall_symbol("P 2y").unwrap(),
        "an explicit Hall origin shift must remain observable"
    );

    assert!(matches!(
        space_group_from_hall_symbol("F m 3 m"),
        Err(SpaceGroupLookupError::UnknownSymbol { .. })
    ));
}

#[test]
fn database_provenance_is_pinned_and_complete() {
    assert_eq!(SPACE_GROUP_DATABASE_PROVENANCE.provider, "moyo");
    assert_eq!(SPACE_GROUP_DATABASE_PROVENANCE.version, "0.15.0");
    assert_eq!(SPACE_GROUP_DATABASE_PROVENANCE.hall_setting_count, 530);
    for number in 1..=230 {
        space_group_by_number(number)
            .unwrap_or_else(|error| panic!("space group {number} did not resolve: {error}"));
    }
}

#[test]
fn every_standard_space_group_has_the_international_crystal_system() {
    let ranges = [
        (1..=2, CrystalSystem::Triclinic),
        (3..=15, CrystalSystem::Monoclinic),
        (16..=74, CrystalSystem::Orthorhombic),
        (75..=142, CrystalSystem::Tetragonal),
        (143..=167, CrystalSystem::Trigonal),
        (168..=194, CrystalSystem::Hexagonal),
        (195..=230, CrystalSystem::Cubic),
    ];
    for (numbers, expected) in ranges {
        for number in numbers {
            let actual = space_group_by_number(number)
                .unwrap_or_else(|error| panic!("space group {number} did not resolve: {error}"))
                .space_group
                .crystal_system();
            assert_eq!(actual, expected, "space group {number}");
        }
    }
}
