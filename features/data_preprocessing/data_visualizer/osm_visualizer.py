import pandas as pd
import folium
from folium.plugins import FastMarkerCluster, HeatMap

# =========================
# 1) 파일 읽기
# =========================
import os

# 현재 실행 중인 파이썬 파일의 폴더 위치를 가져옵니다.
current_path = os.path.dirname(__file__)

# 그 폴더 안에 있는 파일 이름을 합칩니다.
file_path = os.path.join(current_path, "d04_text_meta_2025_10_16.txt")

# PeMS 메타데이터는 보통 탭(tab) 구분 파일이라 sep="\t" 사용
df = pd.read_csv(file_path, sep="\t")

# =========================
# 2) 위도/경도 정리
# =========================
# 숫자로 변환 가능한 값만 남김
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

# 좌표 없는 행 제거
df = df.dropna(subset=["Latitude", "Longitude"]).copy()

# (선택) 캘리포니아 근처만 남기기
# 필요 없으면 이 부분 주석 처리해도 됨
df = df[
    (df["Latitude"] >= 32) & (df["Latitude"] <= 43) &
    (df["Longitude"] >= -125) & (df["Longitude"] <= -114)
].copy()

print("센서 개수:", len(df))

# =========================
# 3) 지도 중심 계산
# =========================
center_lat = df["Latitude"].mean()
center_lon = df["Longitude"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=8,
    tiles="OpenStreetMap"
)

# =========================
# 4) 히트맵 레이어
# =========================
heat_data = df[["Latitude", "Longitude"]].values.tolist()
HeatMap(
    heat_data,
    radius=6,
    blur=4,
    min_opacity=0.2,
    name="Sensor Density"
).add_to(m)

# =========================
# 5) 센서 점 클러스터 레이어
# =========================
# FastMarkerCluster는 대용량(3만 개 내외)에 비교적 잘 버팀
cluster_data = df[["Latitude", "Longitude"]].values.tolist()

FastMarkerCluster(
    cluster_data,
    name="Sensors"
).add_to(m)

# =========================
# 6) 레이어 컨트롤
# =========================
folium.LayerControl().add_to(m)

# =========================
# 7) 저장
# =========================
output_html = "pems_sensors_osm_map.html"
m.save(output_html)

print(f"지도가 저장됨: {output_html}")  