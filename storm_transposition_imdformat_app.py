#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Name: Storm_transposition_imdformat_app.py
Description: Calculation of SPS/PMP from storm contour and catchment shape file.
Author: Vasanthakumar V
Date Created: 2025-05-05
Version: 7.0
"""

import streamlit as st
import geopandas as gpd
import os
from shapely.geometry import Polygon, LineString, MultiLineString, shape
from shapely.affinity import rotate, translate
import numpy as np
from scipy.optimize import minimize
from shapely.ops import unary_union
from scipy.spatial import cKDTree
import rasterio
from rasterio.transform import from_origin
from rasterio.mask import mask
import tempfile
import time
import pydeck as pdk
import pandas as pd
import json
import re

st.set_page_config(layout="wide", page_title="Storm Transposition Optimizer")

# --------------------------
# Helper functions
# --------------------------

def line_to_polygon(geom):
    if isinstance(geom, (LineString, MultiLineString)):
        try:
            return Polygon(geom)
        except:
            return None
    return geom

@st.cache_data(show_spinner=False)
def load_shapefile(uploaded_file):
    return gpd.read_file(uploaded_file)

# --------------------------
# Sidebar Inputs
# --------------------------

st.sidebar.title("Upload Input Shapefiles")

contour_file = st.sidebar.file_uploader("Upload Rainfall Contour Shapefile (.shp, .zip):", type=["zip"])
polygon_file = st.sidebar.file_uploader("Upload Catchment Shapefile (.shp, .zip):", type=["zip"])

resolution = st.sidebar.slider("Interpolation Resolution (m)", 500, 5000, 1000, step=500)
utm_zone = st.sidebar.selectbox("Select UTM Zone", options=[43, 44], index=0, format_func=lambda z: f"UTM Zone {z}")
selected_epsg = 32600 + utm_zone
MMF = st.number_input(
    "Enter Moisture Maximisation factor Factor (MMF) (Only for PMP, IF SPS leave it as 1)",
    min_value=1.0,
    value=1.0,
    step=0.05,
    help="Enter only if project qualifies for PMF, else leave it as 1"
)
# User input for number of top contours
top_n = st.number_input("Enter number of top contours for guesses:", min_value=1, max_value=100, value=3, step=1)

if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = None


if contour_file and polygon_file:
    with st.spinner("Reading shapefiles..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            contour_path = os.path.join(tmpdir, "contour.zip")
            catchment_path = os.path.join(tmpdir, "catchment.zip")

            with open(contour_path, "wb") as f:
                f.write(contour_file.getbuffer())
            with open(catchment_path, "wb") as f:
                f.write(polygon_file.getbuffer())

            contours = gpd.read_file(f"zip://{contour_path}")
            polygon = gpd.read_file(f"zip://{catchment_path}")

    # Reproject
    contours = contours.to_crs(epsg=32643)
    polygon = polygon.to_crs(epsg=32643)
    search_poly = polygon.geometry.iloc[0]

    contours['geometry'] = contours['geometry'].apply(line_to_polygon)
    contours = contours[contours['geometry'].notnull()]
    contours['geometry'] = contours['geometry'].buffer(0)
    contours = contours[contours['geometry'].is_valid]
    R_regions = [(row.geometry, row['CONTOUR']) for _, row in contours.iterrows()]
    contours_union = unary_union([r[0] for r in R_regions])
    

    def cumulative_R(params):
        x, y, theta = params
        moved_poly = rotate(search_poly, theta, origin='centroid', use_radians=False)
        moved_poly = translate(moved_poly, xoff=x, yoff=y)

        if not moved_poly.within(contours_union):
            uncovered_area = moved_poly.difference(contours_union).area
        else:
            uncovered_area = 0

        total_weighted_rainfall, total_area = 0, 0
        for i, (region, R_value) in enumerate(R_regions):
            intersection = moved_poly.intersection(region)
            if intersection.is_empty:
                continue
            net_intersection = intersection
            for j, (inner_region, inner_R_value) in enumerate(R_regions):
                if inner_R_value > R_value:
                    inner_intersection = intersection.intersection(inner_region)
                    if not inner_intersection.is_empty:
                        net_intersection = net_intersection.difference(inner_intersection)
            if net_intersection.is_empty:
                continue
            area = net_intersection.area
            dist_to_region = moved_poly.centroid.distance(region)
            min_other_dist, R_other = float('inf'), None
            for k, (other_region, other_R) in enumerate(R_regions):
                if k == i:
                    continue
                dist = moved_poly.centroid.distance(other_region)
                if dist < min_other_dist:
                    min_other_dist, R_other = dist, other_R
            if R_other is not None and (dist_to_region + min_other_dist) > 0:
                interpolated_rainfall = (
                    (R_value * min_other_dist + R_other * dist_to_region) / (dist_to_region + min_other_dist)
                )
            else:
                interpolated_rainfall = R_value

            total_weighted_rainfall += area * interpolated_rainfall
            total_area += area

        average_rainfall = total_weighted_rainfall / total_area if total_area > 0 else 0
        penalty = 1 * (uncovered_area/1000000)
        return -average_rainfall + penalty
    st.subheader("Optimization")
    if st.button("Run Optimization"):
        with st.spinner("Preparing initial guesses and running optimization..."):

            # Step 1: Add area if not already present
            contours['Area'] = contours.geometry.area

            # Step 2: Sort contours by rainfall then area
            contours_sorted = contours.sort_values(['CONTOUR', 'Area'], ascending=[False, False])

            # Step 3: Take top 3 polygons
            top_contours = contours_sorted.head(top_n)

            # Step 4: Find catchment centroid (fixed)
            centroid_catchment = search_poly.centroid

            # Step 5: Generate initial guesses
            initial_guesses = []
            for _, row in top_contours.iterrows():
                centroid = row.geometry.centroid
                dx = centroid.x - centroid_catchment.x
                dy = centroid.y - centroid_catchment.y
                initial_guesses.append([dx, dy, 0])  # Start with 0° rotation

            # Step 6: Define bounds
            contour_bounds = contours.total_bounds
            polygon_bounds = polygon.total_bounds
            #x_range = contour_bounds[2] - contour_bounds[0]
            #y_range = contour_bounds[3] - contour_bounds[1]
            
            x_range = max(polygon_bounds[2],contour_bounds[2]) - min(polygon_bounds[0],contour_bounds[0])  # max x - min x
            y_range = max(polygon_bounds[3],contour_bounds[3]) - min(polygon_bounds[1],contour_bounds[1]) # max y - min y
            bounds = [(-x_range, x_range), (-y_range, y_range), (-20, 20)]

            # Step 7: Run optimization for each initial guess
            best_result = None
            best_fun = float('inf')

            for idx, guess in enumerate(initial_guesses):
                st.write(f"Trying initial guess {idx+1}: {guess}")
                result = minimize(cumulative_R, guess, bounds=bounds, method='Powell')
                st.write(f"➡️ Result value: {result.fun:.4f}")

                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result

            result = best_result

            # Display final result
            st.success("Optimization Complete")
            best_x, best_y, best_theta = result.x
            st.write(f"✅ Optimal X Offset: {best_x:.2f} m")
            st.write(f"✅ Optimal Y Offset: {best_y:.2f} m")
            st.write(f"✅ Optimal Rotation: {best_theta:.2f}°")
            

            best_poly = rotate(search_poly, best_theta, origin='centroid', use_radians=False)
            best_poly = translate(best_poly, xoff=best_x, yoff=best_y)
            gdf_result = gpd.GeoDataFrame({'geometry': [best_poly]}, crs=selected_epsg)
            st.session_state.optimization_result = {
                "best_x": best_x,
                "best_y": best_y,
                "best_theta": best_theta,
                "best_poly": best_poly,
            }
else:
    st.info("Please upload both contour and catchment shapefiles (.zip format).")

if st.session_state.optimization_result:
    result = st.session_state.optimization_result
    best_x = result["best_x"]
    best_y = result["best_y"]
    best_theta = result["best_theta"]
    best_poly = result["best_poly"]
    gdf_result = gpd.GeoDataFrame({'geometry': [best_poly]}, crs=selected_epsg)
    
    # Move and rotate contours in reverse: from optimized position to original
    def reverse_transform_geometry(geom, x_shift, y_shift, theta_deg, origin_point):
        # Translate back
        geom = translate(geom, xoff=-x_shift, yoff=-y_shift)
        # Rotate back
        geom = rotate(geom, -theta_deg, origin=origin_point, use_radians=False)
        return geom

    # Preserve original contours for map display
    original_contours = contours.copy()

    # Apply reverse transformation to get "shifted storm" visualization
    reversed_contours = contours.copy()
    reversed_contours['geometry'] = reversed_contours['geometry'].apply(
        lambda geom: reverse_transform_geometry(geom, best_x, best_y, best_theta, origin_point=search_poly.centroid)
    )
    
    st.subheader("Design Storm Precipitation")
    # IDW Interpolation
    points, values = [], []
    for _, row in contours.iterrows():
        geom = row.geometry
        rain = row['CONTOUR']
        coords = list(geom.exterior.coords) if geom.geom_type == 'Polygon' else list(geom.coords)
        for coord in coords:
            points.append(coord)
            values.append(rain)
    points, values = np.array(points), np.array(values)

    minx, miny, maxx, maxy = contours.total_bounds
    grid_x, grid_y = np.mgrid[minx:maxx:complex(0, int((maxx - minx)/resolution)),
                              miny:maxy:complex(0, int((maxy - miny)/resolution))]
    grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    tree = cKDTree(points)
    k, p = 3, 2
    dists, idx = tree.query(grid_points, k=k)
    weights = 1 / (dists ** p)
    weights[dists == 0] = 1e12
    interp_vals = np.sum(weights * values[idx], axis=1) / np.sum(weights, axis=1)
    interp_raster = interp_vals.reshape(grid_x.shape)

    transform = from_origin(minx, maxy, resolution, resolution)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        interpolated_path = tmp.name
        with rasterio.open(interpolated_path, 'w', driver='GTiff',
                           height=interp_raster.shape[1], width=interp_raster.shape[0],
                           count=1, dtype='float32', crs=selected_epsg, transform=transform) as dst:
            dst.write(np.flipud(interp_raster.T), 1)

    with rasterio.open(interpolated_path) as src:
        out_image, _ = mask(src, gdf_result.geometry, crop=True, filled=True, nodata=np.nan)

    data = out_image[0]
    mean_rainfall = np.nanmean(data)
    #st.metric("Standard Project Storm (SPS)", f"{mean_rainfall:.2f} mm")
    if MMF > 1.0:
        mean_rainfall *= MMF
        st.metric("Probable Maximum Precipitation (PMP)", f"{mean_rainfall:.2f} mm")
    else:
        st.metric("Standard Project Storm (SPS)", f"{mean_rainfall:.2f} mm")
    #New Block starts .............
    st.subheader("Export Results")
    output_folder = st.text_input("Enter local output folder path to save results (e.g., C:/Users/Name/Documents/Output):")

    if output_folder and os.path.isdir(output_folder):
        try:
            # Save reversed contours
            reversed_contours_path = os.path.join(output_folder, "reversed_contours.shp")
            reversed_contours.to_file(reversed_contours_path)
            
            # Save reversed contours
            original_contours_path = os.path.join(output_folder, "original_contours.shp")
            original_contours.to_file(original_contours_path)

            # Save interpolated raster
            final_raster_path = os.path.join(output_folder, "interpolated_rainfall.tif")
            with rasterio.open(interpolated_path) as src:
                profile = src.profile
                data_to_write = src.read(1)

            with rasterio.open(final_raster_path, 'w', **profile) as dst:
                dst.write(data_to_write, 1)

            st.success(f"✅ Files saved to {output_folder}")
            st.write(f"- Reversed contours: `{reversed_contours_path}`")
            st.write(f"- Interpolated rainfall raster: `{final_raster_path}`")
        except Exception as e:
            st.error(f"Error saving files: {e}")
    elif output_folder:
        st.warning("⚠️ The specified folder path does not exist. Please check the path.")

    # New Block ends .............
        
    # Reproject all to EPSG:4326 for map display
    contours_4326 = contours.to_crs(epsg=4326)
    original_contours_4326 = original_contours.to_crs(epsg=4326)
    reversed_contours_4326 = reversed_contours.to_crs(epsg=4326)
    original_poly_4326 = polygon.to_crs(epsg=4326)
    best_poly_4326 = gdf_result.to_crs(epsg=4326)

    # Add rainfall values as a tooltip-ready column if not present
    if 'CONTOUR' not in contours_4326.columns:
        contours_4326['CONTOUR'] = [0] * len(contours_4326)
    
    
    #Sorting higher rainfall on last and lower rainfall on bottom to avoid overlap
    def detect_and_sort_polygons(gdf):
        # Convert GeoDataFrame to GeoJSON-like dict to access features
        geojson_data = gdf.__geo_interface__['features']
        
        # Convert features into shapely polygons
        polygons = [shape(feature['geometry']) for feature in geojson_data]
        
        # Detect overlaps
        overlaps = []
        for i, poly1 in enumerate(polygons):
            for j, poly2 in enumerate(polygons):
                if i != j and poly1.intersects(poly2):
                    overlaps.append((i, j))  # Store index pairs of overlapping polygons
        
        # Sort the features by area (larger ones first)
        geojson_data.sort(key=lambda feature: shape(feature['geometry']).area, reverse=True)
        
        # Return sorted features only
        return geojson_data
    original_contours_4326 = detect_and_sort_polygons(original_contours_4326)
    reversed_contours_4326 = detect_and_sort_polygons(reversed_contours_4326)
    original_poly_4326 = detect_and_sort_polygons(original_poly_4326)
    
    
    
    # Helper function to generate layers
    def geojson_layer(gdf, fill_color, name="layer"):
        return pdk.Layer(
            "GeoJsonLayer",
            data= gdf,
            get_fill_color=fill_color,
            get_line_color=[0, 0, 0, 255],
            get_line_width=1,
            pickable=True,
            filled=True,
            auto_highlight=True,
            name=name
        )

    # Define layers
    #contour_layer = geojson_layer(contours_4326, [0, 0, 255, 80], "Rainfall Contour")
    original_layer = geojson_layer(original_poly_4326, [0, 255, 0, 150], "Original Catchment")
    #optimized_layer = geojson_layer(best_poly_4326, [255, 0, 0, 150], "Optimized Catchment")

    # Center the map
    centroid = contours_4326.unary_union.centroid
    view_state = pdk.ViewState(
        latitude=centroid.y,
        longitude=centroid.x,
        zoom=8,
        pitch=0
    )
    
    # Display map
    st.subheader("Map Visualization")
    
    
    st.pydeck_chart(pdk.Deck(
        #layers=[contour_layer, original_layer, optimized_layer],
        layers=[
            original_layer,
            geojson_layer(reversed_contours_4326, [255, 100, 0, 60], "Shifted Storm Contours"),
            geojson_layer(original_contours_4326, [100, 100, 255, 60], "Original Rainfall Contours")   
        ],
        initial_view_state=view_state,
        tooltip={"text": "Rainfall: {CONTOUR}"}
    ))
    #Display of shapefiles block END....
    st.markdown("""
    <style>
    .legend-box {
        display: flex;
        gap: 30px;
        margin-top: 10px;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .legend-color {
        width: 20px;
        height: 20px;
        display: inline-block;
    }
    </style>

    <div class="legend-box">
        <div class="legend-item">
            <div class="legend-color" style="background-color: rgba(0, 255, 0, 0.6); border: 1px solid #000;"></div>
            <span>Original Catchment</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: rgba(255, 100, 0, 0.6); border: 1px solid #000;"></div>
            <span>Shifted Storm Contours</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: rgba(100, 100, 255, 0.6); border: 1px solid #000;"></div>
            <span>Original Rainfall Contours</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    
    

