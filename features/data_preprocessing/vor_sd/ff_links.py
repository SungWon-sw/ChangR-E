"""
PeMS 메타데이터의 FF(Freeway-to-Freeway connector) 관측소 Name 필드를 파싱해
"이 지점이 어느 (고속도로,방향) 사슬과 어느 (고속도로,방향) 사슬을 잇는가"를
데이터에서 직접 뽑아낸다.

예: Name="SB 110 TO EB 105"  ->  (110,'S') -- (105,'E')
    Name="90 E - 405 S"      ->  (90,'E')  -- (405,'S')   (어순이 반대인 표기도 있음)
    Name="SB 5 TO WB/SB 2"   ->  (5,'S')   -- (2,'S')     (대각선 고속도로는 방향이
                                                            "EB/NB" 처럼 결합 표기됨 —
                                                            실제 추적 중인 사슬과
                                                            겹치는 글자를 고른다)

좌표 근접성 추측(300m 이내 아무 끝점이나 잇기) 대신, 데이터에 실제로 적힌 사실을
우선 근거로 삼기 위한 모듈. rl_env_voronoi_mw.py._build_road_graph 가 사용한다.
"""

import re

import numpy as np
import pandas as pd

_DIRTOK = r"(?:[NSEW]B?)"
_DIRSET = rf"{_DIRTOK}(?:/{_DIRTOK})*"
_PAT_DIR_FIRST = re.compile(
    rf"^\s*({_DIRSET})\s+(\d+[A-Za-z]?)\s*(?:TO|-)\s*({_DIRSET})\s+(\d+[A-Za-z]?)")
_PAT_NUM_FIRST = re.compile(
    rf"^\s*(\d+[A-Za-z]?)\s+({_DIRSET})\s*(?:TO|-)\s*(\d+[A-Za-z]?)\s+({_DIRSET})")


def _normalize(name):
    # "N/B" 같은 스타일적 슬래시 표기 -> "NB". "EB/NB"(결합방향)는 건드리지 않는다
    # (이 정규식은 '문자 하나 + "/B"' 형태만 잡아서 2글자 방향코드끼리의 결합과
    # 구분된다).
    return re.sub(r"([NSEW])/B\b", r"\1B", str(name).upper())


def _dirset_to_letters(dirset):
    return {tok[0] for tok in dirset.split("/")}


def _resolve(fwy_str, dirset, chain_keys):
    """fwy 번호 + 방향집합을 실제 추적 중인 (fwy,dir) 사슬 키로 해석."""
    try:
        fwy = int(re.sub("[A-Za-z]", "", fwy_str))
    except ValueError:
        return None
    for letter in _dirset_to_letters(dirset):
        key = (fwy, letter)
        if key in chain_keys:
            return key
    return None


def parse_ff_name(name, chain_keys):
    """FF 관측소 Name 하나를 파싱해 (side1, side2) = ((fwy1,dir1),(fwy2,dir2)) 반환.
    실패하거나 양쪽 다 chain_keys 에 없으면 None."""
    s = _normalize(name)
    for pat, order in ((_PAT_DIR_FIRST, "dir_first"), (_PAT_NUM_FIRST, "num_first")):
        mo = pat.match(s)
        if not mo:
            continue
        g1, g2, g3, g4 = mo.groups()
        if order == "dir_first":
            d1, f1, d2, f2 = g1, g2, g3, g4
        else:
            f1, d1, f2, d2 = g1, g2, g3, g4
        side1 = _resolve(f1, d1, chain_keys)
        side2 = _resolve(f2, d2, chain_keys)
        if side1 is not None and side2 is not None and side1 != side2:
            return side1, side2
    return None


def load_ff_links(meta_txt, chain_keys):
    """meta_txt 를 읽어 Type=='FF' 인 관측소 중 Name 이 파싱되고 양쪽 다
    chain_keys(추적 중인 (fwy,dir) 집합)에 속하는 것만 골라, 각 링크의
    (side1, side2, lat, lon) 리스트로 반환한다. 좌표 투영은 호출자가 한다
    (segments/sites 생성 때 쓴 것과 같은 lat0/lon0 기준이어야 정합됨)."""
    m = pd.read_csv(meta_txt, sep="\t")
    ff = m[m.Type == "FF"].dropna(subset=["Latitude", "Longitude"]).copy()
    links = []
    for _, row in ff.iterrows():
        parsed = parse_ff_name(row["Name"], chain_keys)
        if parsed is None:
            continue
        side1, side2 = parsed
        links.append((side1, side2, float(row["Latitude"]), float(row["Longitude"])))
    return links
