import glob
import linecache
import pickle
import sys
from io import BytesIO
from os.path import *

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ship_ice_planner.NRC_wrapper.daq_columns import *
from ship_ice_planner.NRC_wrapper.nrc_config import H_MATRIX_WORLD_TO_CAM
from ship_ice_planner.cost_map import CostMap
from ship_ice_planner.evaluation import PLANNER_DATA_TRANSFORMED
from ship_ice_planner.evaluation.process_daq_data import get_start_end_timesteps
from ship_ice_planner.evaluation.timing_file_parser import get_polygons_from_timing_file
from ship_ice_planner.image_process.display_img_coords import display_img_coords
from ship_ice_planner import PATH_DIR, DEFAULT_CONFIG_FILE
from ship_ice_planner.ship import Ship
from ship_ice_planner.utils.plot import Plot
from ship_ice_planner.utils.utils import DotDict

SHOW_VIDEO = False
OVERLAY_SHIP_POS = True
OVERLAY_PATH = True
OVERLAY_OBS = False
OVERLAY_SHIP_PATH = True
SHOW_COSTMAP = False  # this is quite slow
SAVE_OUT_IMG_EVERY_K_FRAMES = None  # 300

SHIP_XY_COLOR = (255, 255, 255)
PATH_COLOR = (0, 0, 255)
GLOBAL_PATH_COLOR = (255, 255, 0)
OBS_COLOR = (0, 0, 0)

SAVE_OVERLAY_VIDEO_FILENAME = 'overlay_video.mkv'
FRAMES_TO_READ = 2  # this is to speed up the video playback
FRAME_BY_FRAME = False

IMG_BORDERS = [[301, 50], [301, 874], [1622, 874], [1622, 50]]
IMG_Y_OFFSET = 368


def process_overhead_video(video_file_name: str,
                           daq_df: pd.DataFrame = None,
                           show_video: bool = SHOW_VIDEO,
                           overlay_ship_pos: bool = OVERLAY_SHIP_POS,
                           overlay_path: bool = OVERLAY_PATH,
                           overlay_obs: bool = OVERLAY_OBS,
                           overlay_ship_path: bool = OVERLAY_SHIP_PATH,
                           show_costmap: bool = SHOW_COSTMAP,
                           save_overlay_video_filename: str = SAVE_OVERLAY_VIDEO_FILENAME,
                           frames_to_read: int = FRAMES_TO_READ,
                           save_first_frame: bool = False,
                           save_out_img_every_k_frames: int = SAVE_OUT_IMG_EVERY_K_FRAMES):
    if daq_df is None:
        # load in the daq file and get the start and end timesteps
        # otherwise assume it is passed as a dataframe clipped to the start and end timesteps

        csv_file = glob.glob(join(dirname(video_file_name), '*.csv'))
        if len(csv_file):
            daq_df = load_daq_as_df(csv_file[0])
        else:
            exit('No daq data found!')

    start, end = get_start_end_timesteps(daq_df, video_file_name)

    ship_xy = daq_df[POSITION_COLS].to_numpy().reshape(1, -1, 3)[:, :, :2]
    ship_xy_transformed = cv2.perspectiveTransform(ship_xy, H_MATRIX_WORLD_TO_CAM)
    ship_xy_list = []

    cap = cv2.VideoCapture(video_file_name)
    # ret, frame = cap.read()
    # img_boundaries = display_img_coords(frame, num_pts=4); exit()  # useful for determining img boundaries
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame rate', frame_rate)
    print('num frames', num_frames)

    video_size = (IMG_BORDERS[2][0] - IMG_BORDERS[0][0], IMG_BORDERS[1][1] - IMG_BORDERS[0][1] - IMG_Y_OFFSET)

    if show_costmap:
        cfg = DotDict.load_from_file(join(dirname(video_file_name), DEFAULT_CONFIG_FILE))
        costmap = CostMap(ship_mass=cfg.ship.mass,
                          length=cfg.costmap.m,
                          width=cfg.costmap.n,
                          collision_cost_weight=cfg.costmap.collision_cost_weight,
                          boundary_margin=cfg.costmap.margin,
                          scale=cfg.costmap.scale,
                          ice_thickness=0.012,
                          ice_density=991,
                          )

        costmap_min_max = None
        ship = Ship(scale=cfg.costmap.scale, **cfg.ship)

    if save_overlay_video_filename:
        if 'mp4' in save_overlay_video_filename:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')

        out = cv2.VideoWriter(
            join(dirname(video_file_name), save_overlay_video_filename),
            fourcc,
            frame_rate,
            video_size)

        if show_costmap:
            out2 = None

    iteration = 0

    while 1:
        ret, frame = cap.read()
        iteration += 1
        if iteration % frames_to_read != 0:
            continue
        if not ret:
            print('No frames grabbed!')
            break

        polys, timestamp, _ = get_polygons_from_timing_file(dirname(video_file_name),
                                                            step=iteration,
                                                            return_timestamp=True)
        # get the daq_index that corresponds to the current iteration
        daq_index = np.argmin(np.abs(daq_df[TIME_COL] - timestamp))

        if daq_index < start:
            print('Trial not started, skipping... {}'.format(iteration), end='\r')
            continue
        elif daq_index > end:
            print('Trial ended, finishing...')
            break
        elif daq_index > len(daq_df):
            print('No more daq data at index', daq_index)
            break
        print('iteration ........................{} / {}'.format(iteration, num_frames), end='\r')  # progress bar

        # crop to boundaries
        frame = frame[(IMG_BORDERS[0][1] + IMG_Y_OFFSET):IMG_BORDERS[1][1], IMG_BORDERS[0][0]:IMG_BORDERS[2][0], :]

        if save_first_frame:
            cv2.imwrite(join(dirname(video_file_name), 'first_frame.png'), frame)
            save_first_frame = False

        if overlay_ship_pos:
            curr_ship_xy_transformed = ship_xy_transformed[0, daq_index].astype(np.int32) - [0, IMG_Y_OFFSET]
            frame = cv2.circle(frame,
                               tuple(curr_ship_xy_transformed),
                               radius=5,
                               color=SHIP_XY_COLOR,
                               thickness=-1)
            ship_xy_list.append(curr_ship_xy_transformed)
            # add a dot at (0,0) in camera coordinates
            frame = cv2.circle(frame, (0, 0), radius=5, color=(255, 0, 0), thickness=-1)

            if overlay_ship_path:
                # draw the ship path
                cv2.polylines(frame,
                              np.int32([ship_xy_list]),
                              isClosed=False,
                              color=SHIP_XY_COLOR,
                              thickness=2,
                              lineType=cv2.LINE_AA)

        planner_data_world = None  # planner data transformed to world coordinates
        planner_idx = daq_df[PLANNER_COL].iloc[daq_index]

        if overlay_path:
            # get the path
            if planner_idx != -1:
                planner_data_world = pickle.load(open(join(dirname(video_file_name), PLANNER_DATA_TRANSFORMED), 'rb'))
                path = planner_data_world[planner_idx]['path'].reshape(1, -1, 3)[..., :2]
                path_transformed = cv2.perspectiveTransform(path, H_MATRIX_WORLD_TO_CAM) - [[[0, IMG_Y_OFFSET]]]
                # draw the path as polylines
                cv2.polylines(frame,
                              np.int32([path_transformed]),
                              isClosed=False,
                              color=PATH_COLOR,
                              thickness=2,
                              lineType=cv2.LINE_AA)

                if 'global_path' in planner_data_world[planner_idx]:
                    global_path = planner_data_world[planner_idx]['global_path'].reshape(1, -1, 3)[..., :2]
                    global_path_transformed = cv2.perspectiveTransform(global_path, H_MATRIX_WORLD_TO_CAM) - [[[0, IMG_Y_OFFSET]]]
                    # draw the path as polylines
                    cv2.polylines(frame,
                                  np.int32([global_path_transformed]),
                                  isClosed=False,
                                  color=GLOBAL_PATH_COLOR,
                                  thickness=1,
                                  lineType=cv2.LINE_AA)

        if overlay_obs:
            poly_transformed = []
            for poly in polys:
                poly_transformed.append(
                    np.int32(cv2.perspectiveTransform(poly.reshape(1, -1, 2), H_MATRIX_WORLD_TO_CAM) - [[[0, IMG_Y_OFFSET]]])
                )

            cv2.polylines(frame,
                          poly_transformed,
                          isClosed=True,
                          color=OBS_COLOR,
                          thickness=2,
                          lineType=cv2.LINE_AA)

        if show_costmap:
            planner_data = pickle.load(open(join(dirname(video_file_name), PATH_DIR, str(planner_idx) + '.pkl'), 'rb'))
            ship_pos = planner_data['ship_state']
            obs = planner_data['processed_message']['obstacles']

            costmap.update(obs_vertices=obs,
                           ship_pos_y=ship_pos[1] - ship.length,
                           ship_speed=cfg.controller.target_speed,
                           goal=planner_data['goal'])
            if costmap_min_max is None:
                costmap_min_max = (0, costmap.get_costmap_max())  # hack to get colormap range consistent across frames

            plot = Plot(
                costmap=costmap.cost_map,
                obstacles=costmap.all_obstacles,
                path=planner_data['path'],
                global_path=planner_data['global_path'],
                ship_vertices=ship.vertices,
                ship_pos=ship_pos,
                horizon=cfg.optim.horizon * cfg.costmap.scale,
                sim_figsize=None,
                scale=costmap.scale,
                y_axis_limit=min(cfg.plot.y_axis_limit, cfg.costmap.m) * costmap.scale,
                legend=False,
                costmap_min_max=costmap_min_max,
                show=False
            )
            plot.map_fig.canvas.draw()

            # Save the figure to a memory buffer
            buf = BytesIO()
            plot.map_fig.savefig(buf, format='png', dpi=300)
            buf.seek(0)  # Rewind to the beginning of the buffer
            # Convert the buffer into a numpy array
            img_array = np.frombuffer(buf.getvalue(), dtype='uint8')
            costmap_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            plt.close(plot.map_fig)

        if save_overlay_video_filename:
            out.write(frame)

            if show_costmap:
                if out2 is None:
                    out2 = cv2.VideoWriter(
                        join(dirname(video_file_name), 'costmap.' + save_overlay_video_filename.split('.')[1]),
                        fourcc,
                        frame_rate,
                        costmap_img.shape[:2][::-1])
                out2.write(costmap_img)

        if show_video:
            cv2.imshow('frame', frame)

            if show_costmap:
                cv2.imshow('costmap', costmap_img)

        if (
                save_overlay_video_filename and
                save_out_img_every_k_frames is not None and
                iteration % save_out_img_every_k_frames == 0
        ):
            # save as pngs
            cv2.imwrite(join(dirname(video_file_name), 'frame_{}.png'.format(iteration)), frame)
            if show_costmap:
                cv2.imwrite(join(dirname(video_file_name), 'costmap_{}.png'.format(iteration)), costmap_img)

        if show_video and FRAME_BY_FRAME:
            while 1:
                k = cv2.waitKey(30) & 0xff
                if k == 32 or k == 27:
                    break
            if k == 27:
                break
        else:
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == 32:
                while 1:
                    k = cv2.waitKey(30) & 0xff
                    if k == 32:
                        break

    cap.release()
    if save_overlay_video_filename:
        out.release()
        if show_costmap:
            out2.release()

    cv2.destroyAllWindows()
    linecache.clearcache()  # necessary for timing_file_parser.py


if __name__ == '__main__':
    video_file_name = sys.argv[1]
    process_overhead_video(video_file_name)
