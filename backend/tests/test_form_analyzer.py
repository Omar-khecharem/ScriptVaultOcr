"""Tests du moteur d'analyse de formulaire (extraction + validation locales).

Aucune dépendance externe : les items OCR sont fabriqués à la main avec des
boîtes réalistes (label à gauche, valeur à droite sur la même ligne, ou
valeur en dessous).
"""

from __future__ import annotations

import time

import pytest
from scriptvault.form_analyzer import (
    CIN_RE,
    DATE_RE,
    LocalFormAnalyzer,
    demo_items,
)
from scriptvault.schemas import FieldStatus, FormFieldResult, FormSection


def _items(pairs: list[tuple[str, str]], *, confidence: float = 0.96) -> list[dict]:
    """Construit des items OCR ``(label, valeur)`` espacés en colonnes."""
    items: list[dict] = []
    for row, (label, value) in enumerate(pairs):
        y0 = 100 + row * 60
        items.append(
            {
                "text": label,
                "confidence": 0.97,
                "box": [[50, y0], [260, y0], [260, y0 + 34], [50, y0 + 34]],
            }
        )
        items.append(
            {
                "text": value,
                "confidence": confidence,
                "box": [[280, y0], [520, y0], [520, y0 + 34], [280, y0 + 34]],
            }
        )
    return items


def _analyze(pairs, **kwargs):
    analyzer = LocalFormAnalyzer()
    return analyzer.analyze_page(_items(pairs, **kwargs), file_name="test.png")


def _by_key(response):
    return {field.key: field for field in response.fields}


# --------------------------------------------------------------------------- #
# Fiabilité des extractions (anti-fausses données)
# --------------------------------------------------------------------------- #
def test_noise_confidence_gate():
    """Une ligne OCR incertaine ne doit pas l'emporter sur la vraie valeur."""
    y0 = 100
    label = {
        "text": "Nom :",
        "confidence": 0.97,
        "box": [[50, y0], [260, y0], [260, y0 + 34], [50, y0 + 34]],
    }
    value = {
        "text": "Didi",
        "confidence": 0.96,
        "box": [[280, y0], [520, y0], [520, y0 + 34], [280, y0 + 34]],
    }
    noise = {
        "text": "12345",
        "confidence": 0.08,
        "box": [[40, y0 - 20], [180, y0 - 20], [180, y0 + 8], [40, y0 + 8]],
    }
    analyzer = LocalFormAnalyzer(min_item_confidence=0.50)
    fields = _by_key(analyzer.analyze_page([label, value, noise], file_name="t.png"))
    assert fields["nom"].value == "Didi"


def test_noise_gate_keeps_boxless_items():
    """Un item sans boîte (ordre spatial inconnu) reste exploitable."""
    response = LocalFormAnalyzer().analyze_page(
        [
            {"text": "Identifiant :", "confidence": 0.2, "box": []},
            {"text": "514017", "confidence": 0.2, "box": []},
        ],
        file_name="t.png",
    )
    fields = _by_key(response)
    assert fields["identifiant"].value == "514017"


def test_correction_gated_on_digits():
    """O→0 etc. ne s'applique qu'aux valeurs majoritairement numériques."""
    assert LocalFormAnalyzer._correct_ocr_value("cin", "O9728370") == "09728370"
    assert LocalFormAnalyzer._correct_ocr_value("cin", "OMAR097") == "OMAR097"
    assert LocalFormAnalyzer._correct_ocr_value("cin", "K97283ZK") == "K97283ZK"
    assert LocalFormAnalyzer._correct_ocr_value("cin", "OMAR") == "OMAR"


# --------------------------------------------------------------------------- #
# Extraction spatiale
# --------------------------------------------------------------------------- #
def test_demo_scenario_all_valid():
    """La feuille type (Concours→Candidat→Codification) passe en valid."""
    response = _analyze(
        [
            ("Nom du concours :", "Baccalauréat 2026"),
            ("Session :", "Principale"),
            ("Concours :", "Sciences expérimentales"),
            ("Épreuve de :", "Mathématiques"),
            ("Date :", "04/06/2026"),
            ("Durée :", "2h"),
            ("Nom :", "Didi"),
            ("Prénom :", "Mayssa"),
            ("Date & lieu de naissance :", "04/12/2003"),
            ("Établissement d'origine :", "IPEIN"),
            ("N° CIN :", "09728320"),
            ("Série :", "5140"),
            ("Identifiant :", "514017"),
            ("Nombre de cahiers remis :", "2"),
            ("Bloc code-barres / Identifiant d'anonymat :", "5140 514017"),
        ]
    )
    assert response.is_form is True
    assert response.global_confidence == pytest.approx(0.96, abs=0.01)
    assert len(response.fields) == 15
    for field in response.fields:
        assert field.status == FieldStatus.VALID
        assert field.error_message is None


def test_value_below_label_is_found():
    """La valeur peut être sur la ligne suivante (sous le label)."""
    y0 = 100
    items = [
        {
            "text": "Série :",
            "confidence": 0.97,
            "box": [[50, y0], [260, y0], [260, y0 + 30], [50, y0 + 30]],
        },
        {
            "text": "5140",
            "confidence": 0.95,
            "box": [[70, y0 + 60], [150, y0 + 60], [150, y0 + 90], [70, y0 + 90]],
        },
        {
            "text": "Identifiant :",
            "confidence": 0.97,
            "box": [[50, y0 + 140], [260, y0 + 140], [260, y0 + 170], [50, y0 + 170]],
        },
        {
            "text": "514017",
            "confidence": 0.95,
            "box": [[70, y0 + 200], [180, y0 + 200], [180, y0 + 230], [70, y0 + 230]],
        },
    ]
    response = LocalFormAnalyzer().analyze_page(items)
    by_key = _by_key(response)
    assert by_key["serie"].value == "5140"
    assert by_key["identifiant"].value == "514017"


def test_inline_label_value_single_item():
    """Un item unique ``"Nom : Didi"`` est découpé en clé/valeur."""
    response = _analyze([("Nom : Didi", "—"), ("Prénom : Mayssa", "—")])
    by_key = _by_key(response)
    assert by_key["nom"].value == "Didi"
    assert by_key["prenom"].value == "Mayssa"


def test_value_prefers_date_pattern():
    """Pour la date, le candidat matchant le format l'emporte (lieu ignoré)."""
    y0 = 100
    items = [
        {
            "text": "Date & lieu de naissance :",
            "confidence": 0.97,
            "box": [[50, y0], [320, y0], [320, y0 + 30], [50, y0 + 30]],
        },
        {
            "text": "04/12/2003",
            "confidence": 0.95,
            "box": [[60, y0 + 50], [180, y0 + 50], [180, y0 + 80], [60, y0 + 80]],
        },
        {
            "text": "Tunis",
            "confidence": 0.90,
            "box": [[200, y0 + 50], [280, y0 + 50], [280, y0 + 80], [200, y0 + 80]],
        },
    ]
    response = LocalFormAnalyzer().analyze_page(items)
    assert _by_key(response)["date_naissance"].value == "04/12/2003"


def test_non_form_document():
    """Aucun label connu -> is_form False, aucun champ."""
    items = [
        {
            "text": "Rapport de synthèse",
            "confidence": 0.9,
            "box": [[0, 0], [100, 0], [100, 20], [0, 20]],
        },
        {
            "text": "Lorem ipsum dolor sit amet",
            "confidence": 0.9,
            "box": [[0, 40], [300, 40], [300, 60], [0, 60]],
        },
    ]
    response = LocalFormAnalyzer().analyze_page(items)
    assert response.is_form is False
    assert response.fields == []


# --------------------------------------------------------------------------- #
# Validation métier
# --------------------------------------------------------------------------- #
def test_cin_with_letters_is_error():
    response = _analyze([("N° CIN :", "K972832K")])
    cin = _by_key(response)["cin"]
    assert cin.status == FieldStatus.ERROR
    assert "8 chiffres" in (cin.error_message or "")


def test_cin_ocr_confusions_are_corrected():
    """O→0, I→1… : le post-traitement corrige les lectures erronées."""
    response = _analyze([("N° CIN :", "O972832O")])
    cin = _by_key(response)["cin"]
    assert cin.status == FieldStatus.VALID
    assert cin.value == "09728320"


def test_cin_wrong_length_is_error():
    response = _analyze([("N° CIN :", "0972832")])
    assert _by_key(response)["cin"].status == FieldStatus.ERROR


def test_cin_with_spaces_is_accepted():
    response = _analyze([("N° CIN :", "0 9 7 28 320")])
    cin = _by_key(response)["cin"]
    assert cin.status == FieldStatus.VALID
    assert cin.value == "09728320"


def test_cin_nearest_digit_window():
    """Les 8 chiffres les plus proches sont retrouvés dans le bruit OCR."""
    response = _analyze([("N° CIN :", "aa09728320zz")])
    cin = _by_key(response)["cin"]
    assert cin.status == FieldStatus.VALID
    assert cin.value == "09728320"


def test_cin_first_digit_must_be_0_or_1():
    """Règle métier : le CIN tunisien commence par 0 ou 1."""
    for bad in ("89728320", "29728320", "76706906"):
        response = _analyze([("N° CIN :", bad)])
        cin = _by_key(response)["cin"]
        assert cin.status == FieldStatus.ERROR, bad
        assert "0 ou 1" in (cin.error_message or ""), bad
    for good in ("09728320", "19728320"):
        response = _analyze([("N° CIN :", good)])
        assert _by_key(response)["cin"].status == FieldStatus.VALID, good


def test_cin_first_digit_0_or_1_after_correction():
    """Le premier chiffre corrigé (O→0, I→1) satisfait la règle."""
    response = _analyze([("N° CIN :", "O972832O")])
    cin = _by_key(response)["cin"]
    assert cin.status == FieldStatus.VALID
    assert cin.value == "09728320"
    response = _analyze([("N° CIN :", "I972832O")])
    cin = _by_key(response)["cin"]
    assert cin.status == FieldStatus.VALID
    assert cin.value == "19728320"


def test_cin_harvest_prefers_leading_0_or_1():
    """Parmi plusieurs numéros sur la page, le CIN (0/1 en tête) gagne."""
    from scriptvault.form_analyzer import analyze_form_items

    items = [
        {"text": "N° CIN :", "confidence": 0.99, "box": [[0, 0], [80, 0], [80, 12], [0, 12]]},
        {"text": "89728320", "confidence": 0.99, "box": [[120, 0], [220, 0], [220, 12], [120, 12]]},
        {"text": "09728320", "confidence": 0.99, "box": [[120, 40], [220, 40], [220, 52], [120, 52]]},
    ]
    response = analyze_form_items(items, file_name="harvest.png")
    cin = _by_key(response)["cin"]
    assert cin.status == FieldStatus.VALID
    assert cin.value == "09728320"


def test_date_iso_format_is_normalized():
    """« 2003-12-04 » (ISO) → 04/12/2003, accepté sans erreur."""
    response = _analyze([("Date de naissance :", "2003-12-04")])
    field = _by_key(response)["date_naissance"]
    assert field.status == FieldStatus.VALID
    assert field.value == "04/12/2003"


def test_date_year_out_of_range_is_error():
    response = _analyze([("Date de naissance :", "04/12/1940")])
    assert _by_key(response)["date_naissance"].status == FieldStatus.ERROR


def test_date_impossible_calendar_is_error():
    response = _analyze([("Date de naissance :", "31/02/2003")])
    assert _by_key(response)["date_naissance"].status == FieldStatus.ERROR


def test_identifier_serie_mismatch_flags_both():
    """Identifiant 615001 vs Série 5140 -> les DEUX champs passent en rouge."""
    response = _analyze([("Série :", "5140"), ("Identifiant :", "615001")])
    serie = _by_key(response)["serie"]
    identifiant = _by_key(response)["identifiant"]
    assert identifiant.status == FieldStatus.ERROR
    assert serie.status == FieldStatus.ERROR
    assert "doit commencer par la série" in (identifiant.error_message or "")
    assert "ne commence pas par la série" in (serie.error_message or "")


def test_identifier_without_serie_is_valid():
    """Série absente : la règle de cohérence ne s'applique pas."""
    response = _analyze([("Identifiant :", "514017")])
    assert _by_key(response)["identifiant"].status == FieldStatus.VALID


def test_nombre_cahiers_rejects_non_integer():
    response = _analyze([("Nombre de cahiers remis :", "un")])
    assert _by_key(response)["nombre_cahiers"].status == FieldStatus.ERROR


# --------------------------------------------------------------------------- #
# Gabarit réel : Concours / Session / Épreuve / Durée / Anonymat
# --------------------------------------------------------------------------- #
def test_concours_session_epreuve_extracted():
    """Les en-têtes du gabarit (concours, session, épreuve, durée)."""
    response = _analyze(
        [
            ("Nom du concours :", "Baccalauréat 2026"),
            ("Session :", "Principale"),
            ("Concours :", "Sciences expérimentales"),
            ("Épreuve de :", "Mathématiques"),
            ("Durée :", "2h30"),
        ]
    )
    by_key = _by_key(response)
    assert by_key["nom_concours"].value == "Baccalauréat 2026"
    assert by_key["session"].value == "Principale"
    assert by_key["concours"].value == "Sciences expérimentales"
    assert by_key["epreuve"].value == "Mathématiques"
    assert by_key["duree"].value == "2h30"
    assert all(field.status == FieldStatus.VALID for field in response.fields)


def test_duree_longhand_is_warning():
    """Durée en toutes lettres (« deux heures ») : avertissement, pas erreur."""
    response = _analyze([("Durée :", "deux heures")])
    assert _by_key(response)["duree"].status == FieldStatus.WARNING


def test_anonymat_field_accepts_digits():
    response = _analyze([("Bloc code-barres :", "202 601 514")])
    assert _by_key(response)["anonyme"].status == FieldStatus.VALID


# --------------------------------------------------------------------------- #
# Gabarit permanent (mode "include_placeholders")
# --------------------------------------------------------------------------- #
def test_labeled_roi_items_feed_fields_directly():
    """Items pré-étiquetés (lecture par zones) : la valeur du carré EST le
    champ, même à très faible confiance OCR — aucun appariement spatial."""
    items = [
        {
            "label": "nom",
            "text": "Didi",
            "confidence": 0.11,
            "box": [[14, 81], [392, 81], [392, 126], [14, 126]],
        },
        {
            "label": "date_naissance",
            "text": "12/05/2002 Tunis",
            "confidence": 0.55,
            "box": [[14, 240], [380, 240], [380, 300], [14, 300]],
        },
    ]
    response = LocalFormAnalyzer().analyze_page(items, file_name="roi.png")
    by_key = _by_key(response)
    assert by_key["nom"].value
    assert by_key["nom"].status in (FieldStatus.VALID, FieldStatus.WARNING)
    assert by_key["date_naissance"].value.startswith("12/05/2002")
    assert by_key["date_naissance"].status in (
        FieldStatus.VALID,
        FieldStatus.WARNING,
    )


def test_labeled_roi_ignores_garbage_lines():
    """Le chemin ROI ignore les items sans label : une ligne parasite
    (« bruit » TrOCR) ne peut ni remplacer ni polluer les champs lus."""
    items = [
        {
            "label": "cin",
            "text": "09728365",
            "confidence": 0.62,
            "box": [[10, 10], [300, 10], [300, 60], [10, 60]],
        },
        {
            "text": "in the United States's",
            "confidence": 0.90,
            "box": [[80, 620], [600, 620], [600, 660], [80, 660]],
        },
    ]
    response = LocalFormAnalyzer().analyze_page(items, file_name="roi2.png")
    by_key = _by_key(response)
    assert by_key["cin"].value == "09728365"
    assert by_key["cin"].status in (FieldStatus.VALID, FieldStatus.WARNING)
def test_complete_mode_returns_all_gabarit_fields():
    """Mode complet : le gabarit FIXE (11 champs), non lus en statut empty."""
    response = LocalFormAnalyzer().analyze_page(
        _items([("Nom :", "Didi")]),
        file_name="t.png",
        include_placeholders=True,
    )
    keys = {field.key for field in response.fields}
    assert {
        "concours",
        "epreuve",
        "date_concours",
        "duree",
        "nom",
        "prenom",
        "date_naissance",
        "etablissement",
        "cin",
        "serie",
        "identifiant",
    } <= keys
    assert len(response.fields) == 11
    by_key = _by_key(response)
    assert by_key["nom"].status == FieldStatus.VALID
    assert by_key["cin"].status == FieldStatus.EMPTY
    assert by_key["cin"].value == ""
    assert by_key["serie"].status == FieldStatus.EMPTY
    assert "session" not in keys
    assert response.global_confidence == pytest.approx(0.96, abs=0.01)


def test_complete_mode_empty_page_has_placeholders_only():
    """Page sans aucun label : is_form False mais gabarit fixe présent."""
    response = LocalFormAnalyzer().analyze_page(
        [
            {
                "text": "Rapport de synthèse",
                "confidence": 0.9,
                "box": [[0, 0], [100, 0], [100, 20], [0, 20]],
            }
        ],
        include_placeholders=True,
    )
    assert response.is_form is False
    assert len(response.fields) == 11
    assert all(field.status == FieldStatus.EMPTY for field in response.fields)


# --------------------------------------------------------------------------- #
# Gabarit réel : dates en toutes lettres, durées, chiffres espacés par l'OCR
# --------------------------------------------------------------------------- #
def test_french_longhand_date_is_normalized():
    """« Lundi 03 Juin 2024 à 8 H » → 03/06/2024, champ valide."""
    response = _analyze([("Date :", "Lundi 03 Juin 2024 à 8 H")])
    field = _by_key(response)["date_concours"]
    assert field.value == "03/06/2024"
    assert field.status == FieldStatus.VALID


def test_french_longhand_date_with_accents():
    response = _analyze([("Date :", "Mardi 12 décembre 2023")])
    assert _by_key(response)["date_concours"].value == "12/12/2023"


def test_duree_heures_word_is_accepted():
    response = _analyze([("Durée :", "4 Heures")])
    assert _by_key(response)["duree"].status == FieldStatus.VALID


def test_cin_digits_separated_by_spaces():
    """« 0 9 7 28 320 » (OCR pixelisé) → CIN valide 09728320."""
    response = _analyze([("N° CIN :", "0 9 7 28 320")])
    assert _by_key(response)["cin"].status == FieldStatus.VALID


def test_identifier_with_internal_spaces_matches_serie():
    response = _analyze([("Série :", "5140"), ("Identifiant :", "514 017")])
    assert _by_key(response)["identifiant"].status == FieldStatus.VALID


# --------------------------------------------------------------------------- #
# Règles métier complètes (spécification)
# --------------------------------------------------------------------------- #
def test_name_ocr_noise_corrected_via_lexicon():
    """« D1d1 » → corrigé « Didi » grâce au lexique tunisien."""
    response = _analyze([("Nom :", "D1d1")])
    field = _by_key(response)["nom"]
    assert field.value == "Didi"
    assert field.status == FieldStatus.VALID


def test_name_unmatched_ocr_noise_is_error():
    """Bruit OCR sans correspondance lexique : signalé, pas inventé."""
    response = _analyze([("Nom :", "Q1z9x")])
    field = _by_key(response)["nom"]
    assert field.status == FieldStatus.ERROR
    assert "symboles parasites" in (field.error_message or "")


def test_birth_extracts_best_date_and_city():
    """« née le 04/12/1993 à Tnus » → 04/12/1993 · Tnus (lieu structuré)."""
    response = _analyze([("Date & lieu de naissance :", "nee le 04/12/1993 a Tnus")])
    field = _by_key(response)["date_naissance"]
    assert field.value == "04/12/1993 · Tnus"
    assert field.status == FieldStatus.VALID


def test_birth_with_only_date_keeps_single_value():
    response = _analyze([("Date de naissance :", "05.12.2003")])
    field = _by_key(response)["date_naissance"]
    assert field.value == "05/12/2003"
    assert field.status == FieldStatus.VALID


def test_date_string_with_year_is_accepted():
    """Date au format libre (ex. « Lundi 03 Juin 2024 à 8 H ») : acceptée."""
    response = _analyze([("Date :", "Lundi 03 Juin 2024 à 8 H")])
    field = _by_key(response)["date_concours"]
    assert field.value == "03/06/2024"
    assert field.status == FieldStatus.VALID


def test_name_accents_are_accepted():
    response = _analyze([("Prénom :", "Mé hijra'")])
    assert _by_key(response)["prenom"].status == FieldStatus.VALID


def test_etablissement_known_is_valid():
    response = _analyze([("Établissement d'origine :", "IPEIN")])
    assert _by_key(response)["etablissement"].status == FieldStatus.VALID


def test_etablissement_unknown_acronym_is_valid():
    """Un acronyme inconnu mais bien formé reste valide (aucun lexique)."""
    response = _analyze([("Établissement d'origine :", "ZZZZ")])
    assert _by_key(response)["etablissement"].status == FieldStatus.VALID


def test_etablissement_with_digits_is_warning():
    """Chiffres résiduels dans l'acronyme → suspect (ex. TLE1B)."""
    response = _analyze([("Établissement d'origine :", "TLE1B")])
    field = _by_key(response)["etablissement"]
    assert field.status == FieldStatus.WARNING
    assert "suspect" in (field.error_message or "")


def test_etablissement_clean_structure_is_valid():
    """« TLEIB » reste lisible et valide (aucun rapprochement lexique)."""
    response = _analyze([("Établissement d'origine :", "TLEIB")])
    field = _by_key(response)["etablissement"]
    assert field.value == "TLEIB"
    assert field.status == FieldStatus.VALID


def test_etablissement_prepa_short_name_is_valid():
    """IPEIM (prépa de Sfax) est reconnu sans ambiguïté."""
    response = _analyze([("Établissement d'origine :", "IPEIM")])
    assert _by_key(response)["etablissement"].status == FieldStatus.VALID


def test_nombre_cahiers_range_1_to_5():
    assert (
        _by_key(_analyze([("Nombre de cahiers remis :", "7")]))["nombre_cahiers"].status
        == FieldStatus.ERROR
    )
    assert (
        _by_key(_analyze([("Nombre de cahiers remis :", "1")]))["nombre_cahiers"].status
        == FieldStatus.VALID
    )


def test_date_with_dash_separator_is_valid():
    response = _analyze([("Date :", "04-12-2003")])
    assert _by_key(response)["date_concours"].status == FieldStatus.VALID


def test_birth_year_coherent_with_session():
    response = _analyze(
        [("Session :", "2026"), ("Date & lieu de naissance :", "04/12/2003")]
    )
    assert _by_key(response)["date_naissance"].status == FieldStatus.VALID


def test_birth_year_incoherent_with_session_is_error():
    response = _analyze(
        [("Session :", "2026"), ("Date & lieu de naissance :", "04/12/2012")]
    )
    field = _by_key(response)["date_naissance"]
    assert field.status == FieldStatus.ERROR
    assert "incohérente avec la session" in (field.error_message or "")


# --------------------------------------------------------------------------- #
# Série (4 chiffres) / Identifiant (6 chiffres) — longueurs fixes
# --------------------------------------------------------------------------- #
def test_serie_four_digits_is_valid():
    response = _analyze([("Série :", "5401")])
    assert _by_key(response)["serie"].status == FieldStatus.VALID


def test_serie_three_digits_is_valid():
    """« 531 » (3 chiffres) : série officielle acceptée (3 ou 4 chiffres)."""
    response = _analyze([("Série :", "531")])
    assert _by_key(response)["serie"].status == FieldStatus.VALID


def test_serie_two_digits_is_error():
    response = _analyze([("Série :", "54")])
    field = _by_key(response)["serie"]
    assert field.status == FieldStatus.ERROR
    assert "4 chiffres" in (field.error_message or "")


def test_serie_letters_are_error():
    response = _analyze([("Série :", "54AB")])
    assert _by_key(response)["serie"].status == FieldStatus.ERROR


def test_identifiant_six_digits_is_valid():
    response = _analyze([("Identifiant :", "540117")])
    assert _by_key(response)["identifiant"].status == FieldStatus.VALID


def test_identifiant_off_length_is_warning():
    """5 ou 7 chiffres : avertissement, 6 chiffres attendus."""
    for value in ("54011", "5401177"):
        field = _by_key(_analyze([("Identifiant :", value)]))["identifiant"]
        assert field.status == FieldStatus.WARNING
        assert "6 chiffres" in (field.error_message or "")


# --------------------------------------------------------------------------- #
# Relecture ciblée : numéros (CIN/série/identifiant) récupérés sur la page
# --------------------------------------------------------------------------- #
def test_cin_not_on_label_line_is_harvested():
    """Le CIN est écrit dans sa cellule (pas à droite du label) : récolté."""
    y0 = 100
    items = [
        {
            "text": "N° CIN :",
            "confidence": 0.97,
            "box": [[50, y0], [260, y0], [260, y0 + 30], [50, y0 + 30]],
        },
        {
            "text": "0 9 7 28 320",
            "confidence": 0.95,
            "box": [[340, y0 + 140], [520, y0 + 140], [520, y0 + 170], [340, y0 + 170]],
        },
    ]
    response = LocalFormAnalyzer().analyze_page(items)
    cin = _by_key(response)["cin"]
    assert cin.value == "09728320"
    assert cin.status == FieldStatus.VALID


def test_serie_identifiant_harvested_from_same_item():
    """« 5140 540117 » en un seul item OCR : les deux champs sont remplis."""
    y0 = 100
    items = [
        {
            "text": "Série :",
            "confidence": 0.97,
            "box": [[50, y0], [260, y0], [260, y0 + 30], [50, y0 + 30]],
        },
        {
            "text": "Identifiant :",
            "confidence": 0.97,
            "box": [[50, y0 + 200], [260, y0 + 200], [260, y0 + 230], [50, y0 + 230]],
        },
        {
            "text": "5140 5400 17",
            "confidence": 0.95,
            "box": [[340, y0 + 60], [620, y0 + 60], [620, y0 + 90], [340, y0 + 90]],
        },
    ]
    response = LocalFormAnalyzer().analyze_page(items)
    by_key = _by_key(response)
    assert by_key["serie"].value == "5140"
    assert by_key["identifiant"].value == "540017"


def test_serie_not_harvested_when_value_read():
    """La série correctement lue ne doit PAS dépouiller le CIN de la page."""
    response = _analyze([("Série :", "5401"), ("N° CIN :", "09728320")])
    by_key = _by_key(response)
    assert by_key["serie"].value == "5401"
    assert by_key["cin"].value == "09728320"


# --------------------------------------------------------------------------- #
# Dates OCR bruitées : confusions B/O/& + jour inversé
# --------------------------------------------------------------------------- #
def test_birth_noisy_ocr_date_reversed_day():
    """« Bo/09|&00.3 » (OCR brouillon) → 08/09/2003 tout de même."""
    response = _analyze([("Date & lieu de naissance :", "Bo/09|&00.3")])
    field = _by_key(response)["date_naissance"]
    assert field.value == "08/09/2003"
    assert field.status == FieldStatus.VALID


def test_birth_noisy_ocr_date_with_city():
    """« née Bo/09|&00.3 à Tnus » → 08/09/2003 · Tnus (lieu structuré)."""
    response = _analyze([("Date & lieu de naissance :", "nee Bo/09|&00.3 a Tnus")])
    field = _by_key(response)["date_naissance"]
    assert field.value == "08/09/2003 · Tnus"
    assert field.status == FieldStatus.VALID


def test_anonyme_matches_serie_identifiant():
    response = _analyze(
        [
            ("Série :", "5140"),
            ("Identifiant :", "514017"),
            ("Bloc code-barres :", "5140 514017"),
        ]
    )
    assert _by_key(response)["anonyme"].status == FieldStatus.VALID


def test_anonyme_mismatch_serie_identifiant_is_error():
    response = _analyze(
        [
            ("Série :", "5140"),
            ("Identifiant :", "514017"),
            ("Bloc code-barres :", "615 999"),
        ]
    )
    field = _by_key(response)["anonyme"]
    assert field.status == FieldStatus.ERROR
    assert "couple Série/Identifiant" in (field.error_message or "")


# --------------------------------------------------------------------------- #
# Contrôle visuel : zone de signature des enseignants
# --------------------------------------------------------------------------- #
def _signature_items() -> list[dict]:
    return [
        {
            "text": "Zone de signature des enseignants",
            "confidence": 0.97,
            "box": [[100, 600], [500, 600], [500, 630], [100, 630]],
        }
    ]


def test_signature_zone_missing_is_error():
    import numpy as np

    image = np.full((1200, 2000, 3), 255, np.uint8)  # fond vierge
    response = LocalFormAnalyzer().analyze_page(
        _signature_items(), image=image, include_placeholders=True
    )
    zone = _by_key(response)["zone_signature"]
    assert zone.status == FieldStatus.ERROR
    assert "Signature enseignante manquante" in (zone.error_message or "")


def test_signature_zone_present_is_not_error():
    import numpy as np

    image = np.full((1200, 2000, 3), 255, np.uint8)
    image[640:720, 80:640] = (20, 20, 20)  # traits d'encre dans la zone
    analyzer = LocalFormAnalyzer()
    response = analyzer.analyze_page(
        _signature_items(), image=image, include_placeholders=True
    )
    fields = _by_key(response)
    assert "zone_signature" not in fields  # non signalée : absente du gabarit fixe
    assert all(field.status != FieldStatus.ERROR for field in response.fields)


def test_settings_sections_and_canonical_order():
    """Les champs sont triés selon le gabarit : Concours → Candidat → Codif."""
    response = _analyze(
        [
            ("Identifiant :", "514017"),
            ("Série :", "5140"),
            ("Nom :", "Didi"),
            ("Session :", "2026"),
        ]
    )
    keys = [field.key for field in response.fields]
    assert keys.index("session") < keys.index("nom") < keys.index("serie")
    by_key = _by_key(response)
    assert by_key["serie"].section == FormSection.CODIFICATION
    assert by_key["session"].section == FormSection.CONCOURS
    assert by_key["nom"].section == FormSection.CANDIDAT
    assert by_key["session"].section_label == "Concours & Session"
    assert isinstance(by_key["nom"], FormFieldResult)


def test_ocr_correction_applies_only_to_numeric_kinds():
    """Séries et identifiants corrigés (O→0), pas le texte libre."""
    response = _analyze([("Série :", "S140"), ("Épreuve de :", "MathO")])
    by_key = _by_key(response)
    assert by_key["serie"].status == FieldStatus.VALID
    assert by_key["epreuve"].value == "MathO"  # texte : aucune substitution


# --------------------------------------------------------------------------- #
# Classification du risque (seuils de confiance)
# --------------------------------------------------------------------------- #
def test_confidence_warning_range():
    response = _analyze([("Nom :", "Didi")], confidence=0.75)
    nom = _by_key(response)["nom"]
    assert nom.status == FieldStatus.WARNING
    assert nom.error_message is None


def test_confidence_critical_range():
    response = _analyze([("Nom :", "Didi")], confidence=0.60)
    assert _by_key(response)["nom"].status == FieldStatus.ERROR


def test_confidence_69_percent_is_warning_not_error():
    """Seuil critique abaissé à 65 % : une lecture à 69 % reste un warning."""
    response = _analyze([("Nom :", "EPbejaeui")], confidence=0.69)
    nom = _by_key(response)["nom"]
    assert nom.status == FieldStatus.WARNING
    assert "Confiance OCR critique" not in (nom.error_message or "")


def test_confidence_exactly_at_thresholds():
    analyzer = LocalFormAnalyzer()
    valid = analyzer.analyze_page(_items([("Nom :", "Didi")], confidence=0.85))
    assert _by_key(valid)["nom"].status == FieldStatus.VALID
    critical = analyzer.analyze_page(_items([("Nom :", "Didi")], confidence=0.70))
    assert _by_key(critical)["nom"].status == FieldStatus.WARNING


# --------------------------------------------------------------------------- #
# Performance (budget < 30 ms) et contrat
# --------------------------------------------------------------------------- #
def test_processing_time_budget():
    """Le post-traitement d'un formulaire type reste < 30 ms."""
    analyzer = LocalFormAnalyzer()
    started = time.perf_counter()
    for _ in range(20):
        analyzer.analyze_page(demo_items())
    per_doc_ms = (time.perf_counter() - started) * 1000.0 / 20
    assert per_doc_ms < 30.0


def test_analyzed_form_contract():
    response = _analyze([("Nom :", "Didi")])
    assert response.file_name == "test.png"
    assert response.processing_time_ms >= 0.0
    field = response.fields[0]
    assert field.bounding_box == [[280, 100], [520, 100], [520, 134], [280, 134]]
    assert field.label == "Nom"


# --------------------------------------------------------------------------- #
# Regex exposées
# --------------------------------------------------------------------------- #
def test_public_regex():
    assert CIN_RE.match("09728320")
    assert not CIN_RE.match("O972832O")
    assert DATE_RE.match("04/12/2003")
    assert DATE_RE.match("04.12.2003")
    assert not DATE_RE.match("32/13/2003")
    assert not DATE_RE.match("04/12/03")
    assert not DATE_RE.match("O4/12/2003")


# --------------------------------------------------------------------------- #
# Document réel « Eya Elloumi » (lot TIF, OCR 73–79 %) — 7 alertes → vertes
# --------------------------------------------------------------------------- #
def test_eya_elloumi_document_all_green():
    """La copie Eya est lue sans fausse donnée : date + durée en clair OK,
    nom/prénom corrigés, naissance partielle en avertissement, identifiant
    récupéré sur la page malgré le fragment de label parasitaire.
    """
    pairs = [
        ("Concours :", "Physique & Chimie"),
        ("Epreuve de :", "Physique"),
        ("Date :", "Lundi 03 Juin 2024 a 8 H"),
        ("Durée :", "4 Heures"),
        ("Nom :", "Elloom"),
        ("Prénom :", "二"),
        ("Date & lieu de naissance :", "A2/2oo3.Sax"),
        ("Établissement d'origine :", "IETS"),
        ("N° CIN :", "11106906"),
        ("Série :", "531"),
        ("Identifiant :", "N0m 8re de"),
    ]
    items = _items(pairs)
    # Le prénom manuscrit « Eya. » est une unité OCR à part, sous la ligne.
    y_prenom = 100 + 5 * 60
    items.append(
        {
            "text": "Eya.",
            "confidence": 0.55,
            "box": [
                [280, y_prenom + 50],
                [520, y_prenom + 50],
                [520, y_prenom + 84],
                [280, y_prenom + 84],
            ],
        }
    )
    y_ident = 100 + 10 * 60
    items.append(
        {
            "text": "531 531007",
            "confidence": 0.9,
            "box": [
                [280, y_ident + 80],
                [520, y_ident + 80],
                [520, y_ident + 114],
                [280, y_ident + 114],
            ],
        }
    )
    items.append(
        {
            "text": "622 / 6003 45",
            "confidence": 0.88,
            "box": [
                [650, y_ident + 140],
                [900, y_ident + 140],
                [900, y_ident + 174],
                [650, y_ident + 174],
            ],
        }
    )
    by_key = _by_key(LocalFormAnalyzer().analyze_page(items, file_name="eya.tif"))
    assert by_key["date_concours"].value == "03/06/2024"
    assert by_key["date_concours"].status == FieldStatus.VALID
    assert by_key["duree"].value == "4 Heures"
    assert by_key["duree"].status == FieldStatus.VALID
    assert by_key["nom"].value == "Elloom"
    assert by_key["nom"].status == FieldStatus.VALID
    assert by_key["prenom"].value == "Eya"
    assert by_key["prenom"].status == FieldStatus.VALID
    assert by_key["etablissement"].status == FieldStatus.VALID
    assert by_key["cin"].value == "11106906"
    assert by_key["cin"].status == FieldStatus.VALID
    assert by_key["serie"].value == "531"
    assert by_key["serie"].status == FieldStatus.VALID
    assert by_key["identifiant"].value == "531007"
    assert by_key["identifiant"].status == FieldStatus.VALID
    nai = by_key["date_naissance"]
    assert "2003" in nai.value
    assert nai.status != FieldStatus.ERROR


def test_cjk_glyph_is_parasite_noise():
    """Un glyphe non latin (« 二 ») n'est jamais une valeur de champ."""
    from scriptvault.form_analyzer import _is_noise_value

    assert _is_noise_value("二") is True


def test_date_longhand_without_accent():
    """« a 8 H » sans accent : la date reste normalisée en JJ/MM/AAAA."""
    response = _analyze([("Date :", "Lundi 03 Juin 2024 a 8 H")])
    field = _by_key(response)["date_concours"]
    assert field.value == "03/06/2024"
    assert field.status == FieldStatus.VALID
