import pytest
import pandas as pd
from prereq_module import build_graph, build_catalog, is_eligible, get_eligible_courses, get_catalog_df

@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {'course_id': 'CS101',   'subject_code': 'COMP SCI', 'number': '101', 'name': 'Intro CS',    'credits': 3, 'level': 100},
        {'course_id': 'CS201',   'subject_code': 'COMP SCI', 'number': '201', 'name': 'Data Struct', 'credits': 3, 'level': 200},
        {'course_id': 'CS301',   'subject_code': 'COMP SCI', 'number': '301', 'name': 'Algorithms',  'credits': 3, 'level': 300},
        {'course_id': 'MATH101', 'subject_code': 'MATH',     'number': '101', 'name': 'Calculus I',  'credits': 4, 'level': 100},
        {'course_id': 'MATH201', 'subject_code': 'MATH',     'number': '201', 'name': 'Calculus II', 'credits': 4, 'level': 200},
        {'course_id': 'MATH301', 'subject_code': 'MATH',     'number': '301', 'name': 'Linear Alg',  'credits': 3, 'level': 300},
        {'course_id': 'PHYS101', 'subject_code': 'PHYSICS',  'number': '101', 'name': 'Mechanics',   'credits': 4, 'level': 100},
        {'course_id': 'PHYS201', 'subject_code': 'PHYSICS',  'number': '201', 'name': 'E&M',         'credits': 4, 'level': 200},
    ])

@pytest.fixture
def manual_exceptions():
    return {
        'CS201':   ['MATH101'],
        'PHYS201': ['MATH201'],
    }

@pytest.fixture
def graph(sample_df, manual_exceptions):
    return build_graph(sample_df, manual_exceptions=manual_exceptions)

def test_root_courses_have_no_prereqs(graph):
    assert graph['CS101']   == []
    assert graph['MATH101'] == []
    assert graph['PHYS101'] == []

def test_within_subject_chain(graph):
    assert 'CS101' in graph['CS201']
    assert 'MATH101' in graph['MATH201']

def test_300_level_requires_200(graph):
    assert 'CS201' in graph['CS301']
    assert 'MATH201' in graph['MATH301']

def test_cross_subject_exception_added(graph):
    assert 'MATH101' in graph['CS201']

def test_cross_subject_phys(graph):
    assert 'MATH201' in graph['PHYS201']

def test_eligible_no_prereqs(graph):
    assert is_eligible([], 'CS101', graph) is True

def test_eligible_prereqs_met(graph):
    transcript = [
        {'course_id': 'CS101',   'grade': 'B', 'completed': True},
        {'course_id': 'MATH101', 'grade': 'A', 'completed': True},
    ]
    assert is_eligible(transcript, 'CS201', graph) is True

def test_not_eligible_missing_prereq(graph):
    transcript = [
        {'course_id': 'CS101', 'grade': 'B', 'completed': True},
    ]
    assert is_eligible(transcript, 'CS201', graph) is False

def test_not_eligible_incomplete_course(graph):
    transcript = [
        {'course_id': 'CS101',   'grade': 'W', 'completed': False},
        {'course_id': 'MATH101', 'grade': 'A', 'completed': True},
    ]
    assert is_eligible(transcript, 'CS201', graph) is False

def test_eligible_chain_300_level(graph):
    transcript = [
        {'course_id': 'CS101',   'grade': 'A', 'completed': True},
        {'course_id': 'CS201',   'grade': 'B', 'completed': True},
        {'course_id': 'MATH101', 'grade': 'A', 'completed': True},
    ]
    assert is_eligible(transcript, 'CS301', graph) is True

def test_unknown_course_is_eligible(graph):
    assert is_eligible([], 'UNKNOWN999', graph) is True

def test_empty_transcript_only_100_level(graph):
    eligible = get_eligible_courses([], graph)
    for cid in eligible:
        assert graph.get(cid, []) == []

def test_full_transcript_unlocks_higher_levels(graph):
    all_courses = list(graph.keys())
    transcript  = [{'course_id': cid, 'grade': 'A', 'completed': True} for cid in all_courses]
    eligible    = get_eligible_courses(transcript, graph)
    assert set(eligible) == set(all_courses)

def test_catalog_df_has_required_columns(sample_df, manual_exceptions):
    catalog = build_catalog(sample_df, manual_exceptions=manual_exceptions)
    df      = get_catalog_df(catalog)
    required = {'course_id', 'subject', 'level', 'name', 'credits', 'prerequisite_ids'}
    assert required.issubset(set(df.columns))

def test_catalog_df_row_count(sample_df, manual_exceptions):
    catalog = build_catalog(sample_df, manual_exceptions=manual_exceptions)
    df      = get_catalog_df(catalog)
    assert len(df) == len(sample_df)