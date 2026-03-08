# src/app.py
import os
import sys
import logging
import numpy as np
from tqdm import tqdm
from config import load_config
from utils import get_logger, ProgressTracker, read_video, save_video
from tracking import CourtDetection, PlayerTracker, BallTracker
from shot_detection import ShotDetector
from analysis import Dashboard, Metrics, SideCourt
import supervision as sv

def main(max_steps=None):

    # ╔═══════════════════════════════════════════════════════════════════════════╗
    # ║   
    # ║   ██████╗  █████╗ ██████╗ ███╗   ███╗██╗███╗   ██╗████████╗ ██████╗ ███╗   ██╗  
    # ║   ██╔══██╗██╔══██╗██╔══██╗████╗ ████║██║████╗  ██║╚══██╔══╝██╔═══██╗████╗  ██║  
    # ║   ██████╔╝███████║██║  ██║██╔████╔██║██║██╔██╗ ██║   ██║   ██║   ██║██╔██╗ ██║  
    # ║   ██╔══██╗██╔══██║██║  ██║██║╚██╔╝██║██║██║╚██╗██║   ██║   ██║   ██║██║╚██╗██║  
    # ║   ██████╔╝██║  ██║██████╔╝██║ ╚═╝ ██║██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚████║  
    # ║   ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝  
    # ║                                                                            
    # ║         🎾 AI-Powered Badminton Performance Analysis System 🎾            ║
    # ║             ...Version 2.1.0 | © 2024 SportsAI Analytics...               ║
    # ╚═══════════════════════════════════════════════════════════════════════════╝

    logger = get_logger(__name__, log_file='logs/app.log', level=logging.DEBUG)

    total_steps = 9 # Total number of pipeline steps 
    progress = ProgressTracker(total_steps=total_steps, logger=logger)
    progress.set_max_steps(max_steps)
    
    try:
        # region 🚀 APPLICATION INITIALIZATION
        # ╔════════════════════════════════════════════════════════════════╗
        # ║                 🚀 STEP-1:APPLICATION INITIALIZATION           ║
        # ╚════════════════════════════════════════════════════════════════╝
        progress.begin_step("Application Initialization")
        logger.info("🚀 Starting Badminton Analysis System")
        logger.debug(f"Python {'.'.join(map(str, sys.version_info[:3]))}")
        progress.end_step()

        # endregion

        # region ⚙️ CONFIGURATION LOADING
        # ╔════════════════════════════════════════════════════════════════╗
        # ║                 ⚙️ STEP-2:CONFIGURATION LOADING                 ║
        # ╚════════════════════════════════════════════════════════════════╝
        progress.begin_step("Configuration Loading")
        with logger.context("Loading config files"):
            config = load_config()
        progress.end_step(success=True, message=f"Version: {config.get('version', '1.0')}")

        # endregion

        # region 🎥 VIDEO PROCESSING
        # ╔════════════════════════════════════════════════════════════════╗
        # ║                  🎥 STEP-3:VIDEO PROCESSING                    ║
        # ╚════════════════════════════════════════════════════════════════╝
        progress.begin_step("Video Processing")
        with logger.context("Decoding video"):
            frames, fps, width, height = read_video(
                path=config["video"]["input_video"],
                max_frames=config["video"].get("max_frames"),
                show_progress=config["video"].get("show_progress")
            )
            logger.info(f"📹 Loaded {len(frames)} frames @ {fps}fps ({width}x{height})")
        progress.end_step()

        # endregion

        # region 🏟️ COURT DETECTION
        # ╔════════════════════════════════════════════════════════════════╗
        # ║                   🏟️  STEP-4:COURT DETECTION                   ║
        # ╚════════════════════════════════════════════════════════════════╝
        progress.begin_step("Court Detection")
        court_detector = CourtDetection(config, logger)

        if court_detector.court_detection_exists():
            logger.info("Loading existing court detection...")
            court_keypoints = court_detector.load_court_keypoints()
            if not court_keypoints:
                raise ValueError("Failed to load existing court keypoints")
            logger.info(f"🏟️ Loaded {len(court_keypoints)} keypoints")
        else:
            with logger.context("Detecting court features"):
                court_keypoints = court_detector.get_court_keypoints(frames)
                if not court_keypoints:
                    raise ValueError("No court keypoints detected")
                logger.info(f"🏟️ Found {len(court_keypoints)} keypoints")

            with logger.context("Generating court visualization"):
                court_detector.save_video_with_keypoints(frames, court_keypoints)

        progress.end_step()

        # endregion

        # region 👥 PLAYER TRACKING
        # ╔════════════════════════════════════════════════════════════════╗
        # ║                     👥 STEP-5:PLAYER TRACKING                  ║
        # ╚════════════════════════════════════════════════════════════════╝
        progress.begin_step("Player Tracking")
        # Initialize PlayerTracker
        player_tracker = PlayerTracker(
            model_path=config["players"]["players_tracker_model"],
            batch_size=config["players"].get("players_tracker_batch_size", 4),
            frame_rate=fps,
        )
        # Set court zone for the tracker
        player_tracker.set_court_zone(court_keypoints)

        with logger.context("Tracking players"):
            # Check for existing tracking data
            tracking_data_path = config["players"]["players_tracking_path"]
            if os.path.exists(tracking_data_path):
                tracking_data = player_tracker.load_tracking_data(tracking_data_path)
                players_detection = player_tracker.update_tracker(tracking_data) if tracking_data else None
            else:
                players_detection = None
            # If no tracking data, run tracker
            if not players_detection:
                players_detection = player_tracker.run_tracker(frames.copy())
                if not os.path.exists(tracking_data_path):
                    player_tracker.save_tracking_results(players_detection, tracking_data_path)
            
            # Annotate frames 
            annotated_frames = []
            annotation_config = config["players"]["annotation"]
            with tqdm(zip(frames, players_detection), total=len(frames), 
                     disable=not config["video"]["show_progress"]) as pbar:
                for frame, players in pbar:
                    annotated_frames.append(players.annotate_frame(
                        frame,
                        generate_masks=annotation_config.get("generate_masks", True),
                        mask_color=getattr(sv.Color, annotation_config.get("mask_color", "RED")),
                        mask_color_lookup=sv.ColorLookup.INDEX,
                        label_color=getattr(sv.Color,annotation_config.get("label_color", "BLUE")),
                        label_text_scale=annotation_config.get("label_text_scale", 0.7),
                        ellipse_thickness=annotation_config.get("ellipse_thickness", 2),
                        ellipse_color=getattr(sv.Color, annotation_config.get("ellipse_color", "BLUE")),
                        show_confidence=annotation_config.get("show_confidence", True)
                    ))
            logger.info(f"👥 Tracked {len(players_detection)} player positions")
        progress.end_step()

        # endregion

        # region 🎾 BALL TRACKING
        # ╔════════════════════════════════════════════════════════════════╗
        # ║                     🎾 STEP-6:BALL TRACKING                    ║
        # ╚════════════════════════════════════════════════════════════════╝
        progress.begin_step("Ball Tracking")

        # Initialize BallTracker
        ball_tracker = BallTracker(
            tracking_model_path=config["ball"]["tracker_model"],
            inpaint_model_path=config["ball"]["inpaint_model"],
            batch_size=config["ball"].get("tracker_batch_size", 4),
            frame_rate=fps
        )

        with logger.context("Tracking ball"):
            # Check for existing tracking data
            tracking_data_path = config["ball"]["tracking_path"]
            if os.path.exists(tracking_data_path):
                tracking_data = ball_tracker.load_tracking_data(tracking_data_path)
                ball_detections = ball_tracker.update_tracker(tracking_data) if tracking_data else None
            else:
                ball_detections = None
            
            # If no tracking data, run tracker
            if not ball_detections:
                ball_detections = ball_tracker.run_tracker(frames.copy())
                if ball_detections and not os.path.exists(tracking_data_path):
                    if not ball_tracker.save_tracking_results(ball_detections, tracking_data_path):
                        logger.warning("⚠️ Failed to save tracking results")
            
            if not ball_detections:
                raise ValueError("No ball detections generated")

            # Annotate frames
            annotated_frames = ball_tracker.annotate_frames(
                frames=annotated_frames,
                ball_detections=ball_detections,
                trail_length=config["ball"].get("trail_length", 8),
                ball_color=config["ball"].get("ball_color", "YELLOW"),
                trail_color=config["ball"].get("trail_color", "CYAN"),
                ball_radius=config["ball"].get("ball_radius", 5)
            )
                        
            visible_balls = len([b for b in ball_detections if b.visibility == 1])
            logger.info(f"🎾 Tracked {visible_balls} visible ball positions")
        progress.end_step()

        # endregion

        # region 🏸 SHOT DETECTION
        # ╔════════════════════════════════════════════════════════════════╗
        # ║                  🏸 STEP-7:SHOT DETECTION                      ║
        # ╚════════════════════════════════════════════════════════════════╝
        progress.begin_step("Shot Detection")

        with logger.context("Detecting shots"):
            shot_detector = ShotDetector(
                api_key=config["shot_detection"]["api_key"],
                project_name=config["shot_detection"]["project_name"],
                version=config["shot_detection"].get("version", 1),
            )

            # Run detection pipeline (returns annotated frames + shots)
            shots, annotated_frames = shot_detector.run(
                frames=annotated_frames, 
                track_json=config["players"]["players_tracking_path"],
                conf_threshold=config["shot_detection"].get("conf_threshold", 0.3),)

            if not shots:
                logger.warning("⚠️ No shots detected in the video")
            else:
                logger.info(f"🏸 Detected {len(shots)} shots")
                # Build fast lookup: frame_idx -> primary_shot dict
                # shots is a sparse list of {"frame_index": i, "primary_shot": {...}}
                shots_by_frame = {s["frame_index"]: s["primary_shot"] for s in shots} if shots else {}               

        progress.end_step()
        # endregion


        # region 🏸 SIDE COURT VISUALIZATION
        # ╔════════════════════════════════════════════════════════════════╗
        # ║                  🏸 STEP-8:SIDE COURT VISUALIZATION            ║
        # ╚════════════════════════════════════════════════════════════════╝
        progress.begin_step("Side Court Visualization")
        
        # Initialize side court
        side_court = SideCourt(
            width = width,
            height = height,
            position = config["side_court"].get("position", "top_right"),
            scale_factor = config["side_court"].get("scale_factor", 1.0),
            alpha = config["side_court"].get("alpha", 0.5)
        )

        with logger.context("Adding side court visualization"):
            sidecourt_frames = []
            for frame, players, ball in zip(
                annotated_frames,
                players_detection,
                ball_detections,
            ):
                # Add side court visualization to each frame
                frame_with_sidecourt = side_court.draw_court(
                    frame=frame,
                    keypoints_detection=court_keypoints,
                    players_detection=players,
                    ball_detection=ball
                )
                sidecourt_frames.append(frame_with_sidecourt)
            
            logger.info(f"🏸 Added side court to {len(sidecourt_frames)} frames")
        progress.end_step()

        logger.info(f"Successfully processed {len(sidecourt_frames)} frames with side court overlay")
        # endregion

        # region 📊 DASHBOARD GENERATION
        # ╔════════════════════════════════════════════════════════════════╗
        # ║                 📊 STEP-9:DASHBOARD GENERATION                 ║
        # ╚════════════════════════════════════════════════════════════════╝
        progress.begin_step("Dashboard Generation")
        with logger.context("Creating real-time analytics dashboard"):
            dashboard = Dashboard(width, height, fps, court_keypoints, config)
            final_frames = []
            
            # Configuration
            save_interval = config["dashboard"].get("metrics_save_interval", 1)
            batch_size = 50  # Process 50 frames at a time
            
            total_frames = len(sidecourt_frames)
            
            for batch_start in range(0, total_frames, batch_size):
                batch_end = min(batch_start + batch_size, total_frames)
                batch_frames = []
                
                logger.debug(f"Processing batch: frames {batch_start} to {batch_end-1}")
                
                for frame_idx in range(batch_start, batch_end):
                    try:
                        frame = sidecourt_frames[frame_idx]
                        players = players_detection[frame_idx]
                        ball = ball_detections[frame_idx]
                        shot_event = shots_by_frame.get(frame_idx, {})
                        
                        # Determine if we should save metrics this frame
                        save_metrics = (frame_idx % save_interval == 0)
                        
                        final_frame = dashboard.draw(
                            frame,
                            players_detection=players,
                            ball_detection=ball,
                            shot_data=shot_event,
                            save_metrics=save_metrics
                        )
                        batch_frames.append(final_frame)
                        
                        # Log progress periodically
                        if frame_idx % 10 == 0:
                            logger.debug(f"Processed frame {frame_idx}/{total_frames}")
                            
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                        # Fallback: add original frame if dashboard fails
                        batch_frames.append(frame)
                        continue
                
                # Extend the final frames list with this batch
                final_frames.extend(batch_frames)
                
                # Optional: Clear memory if needed
                del batch_frames
                # gc.collect()  # Uncomment if you need to force garbage collection

            # Save final metrics summary
            try:
                dashboard.save_final_metrics()
                logger.info("Saved final metrics summary")
            except Exception as e:
                logger.error(f"Error saving final metrics: {str(e)}")

            logger.info(f"📊 Dashboard generated - Processed {len(final_frames)} frames")
        progress.end_step()
        # endregion

        # region 🏆 FINALIZATION & SUMMARY
        # ╔════════════════════════════════════════════════════════════════╗
        # ║                🏆 STEP-10:FINALIZATION & SUMMARY                ║
        # ╚════════════════════════════════════════════════════════════════╝
        progress.begin_step("Finalization")
        with logger.context("Saving output video"):
            if not sidecourt_frames:
                logger.warning("⚠️ No frames available for saving")
            else:
                save_video(
                    frames=final_frames,
                    path=config["video"]["final_output"],
                    fps=fps,
                    width=width,
                    height=height,
                    show_progress=config["video"]["show_progress"]
                )
                logger.info(f"💾 Saved final output to {config['video']['final_output']}")
        progress.summary()
        logger.info("🏆 Analysis completed successfully")

        # endregion

    except Exception as e:
        logger.exception(f"💥 Application failed: {str(e)}")
        progress.summary()
        sys.exit(1)

if __name__ == "__main__":
    main(max_steps=None)  # Set max_steps to the number of steps you want to run, or None to run all steps