from classification_backend.dose_response.channel_block_generator import ChannelBlockGenerator, ChannelIC50Inputs


def test_generate_single_record_matches_expected_schema():
    generator = ChannelBlockGenerator(ChannelIC50Inputs(herg_ic50_nm=100.0, nav_ic50_nm=200.0, cav_ic50_nm=300.0))
    record = generator.to_ord_payload(100.0)

    assert set(record.keys()) == {"concentration", "herg_block", "nav_block", "cav_block"}
    assert record["concentration"] == 100.0
    assert record["herg_block"] == 50.0


def test_generate_series_produces_five_levels():
    generator = ChannelBlockGenerator({"herg_ic50_nm": 100.0, "nav_ic50_nm": 200.0, "cav_ic50_nm": 300.0})
    frame = generator.block_profile(reference_concentration_nm=10.0)

    assert len(frame) == 5
    assert list(frame["multiple"]) == [0.01, 0.1, 1.0, 10.0, 100.0]
