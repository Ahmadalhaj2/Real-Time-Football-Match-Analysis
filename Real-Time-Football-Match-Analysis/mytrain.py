import os
import cv2
import numpy as np
import yaml
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import supervision as sv
import torch
import matplotlib.pyplot as plt
from supervision import Detections, ByteTrack

class PlayerBallTracker:
    """A complete system for training and using YOLOv8 models to track football players and balls."""
    
    def __init__(self, model_path=None, project_dir="player_ball_tracker", model_name="yolov8n.pt"):
        self.project_dir = project_dir
        self.model = None
        self.model_name = model_name # Default YOLOv8 Nano model for training
        self.team_colors = {
            "barcelona": (255, 0, 0),    # Blue for Barcelona (BGR)
            "real_madrid": (255, 255, 255),  # White for Real Madrid (BGR)
            "ball": (0, 255, 255),         # Yellow for ball (BGR)
            "unknown": (0, 255, 0)         # Green for unknown (BGR)
        }
        os.makedirs(project_dir, exist_ok=True)
        
        if model_path:
            self.load_model(model_path)
    
    def create_dataset_yaml(self, dataset_path, train_dir_name="train", val_dir_name="val", 
                            player_names=None, include_ball=True, team_mapping=None):
        """
        Create YAML configuration file for the custom dataset, compatible with YOLOv8.
        
        Args:
            dataset_path (str): The root directory of the dataset (e.g., 'dataset/').
            train_dir_name (str): The name of the training data directory within dataset_path (default: 'train').
            val_dir_name (str): The name of the validation data directory within dataset_path (default: 'val').
            player_names (list): List of player names that define the classes.
            include_ball (bool): Whether to include 'ball' as a class.
            team_mapping (dict): Dictionary mapping player names to their teams for consistent coloring.
        
        Returns:
            tuple: Path to the generated YAML file and the class dictionary.
        """
        if player_names is None:
            raise ValueError("Player names must be provided to create dataset.yaml.")
        
        class_names = player_names.copy()
        if include_ball and "ball" not in class_names:
            class_names.append("ball")
        
        class_dict = {i: name for i, name in enumerate(class_names)}
        
        # Ensure paths are absolute for YOLOv8 training command, but relative for internal YAML structure
        dataset_config = {
            'path': str(Path(dataset_path).resolve()), 
            'train': os.path.join(train_dir_name, "images"),
            'val': os.path.join(val_dir_name, "images"),
            'names': class_dict
        }
        
        yaml_path = os.path.join(self.project_dir, "dataset.yaml")
        with open(yaml_path, 'w') as file:
            yaml.dump(dataset_config, file, sort_keys=False) 
            
        if team_mapping:
            self.save_team_mapping(team_mapping)
        
        print(f"Dataset YAML created at: {yaml_path}")
        return yaml_path, class_dict
    
    def save_team_mapping(self, team_mapping):
        """Save player-to-team mapping for visualization."""
        team_mapping_path = os.path.join(self.project_dir, "team_mapping.yaml")
        with open(team_mapping_path, 'w') as file:
            yaml.dump(team_mapping, file)
        print(f"Team mapping saved to {team_mapping_path}")
    
    def load_team_mapping(self):
        """Load player-to-team mapping."""
        team_mapping_path = os.path.join(self.project_dir, "team_mapping.yaml")
        if os.path.exists(team_mapping_path):
            with open(team_mapping_path, 'r') as file:
                return yaml.safe_load(file)
        return {}
    
    def train(self, dataset_yaml, epochs=50, batch_size=8, imgsz=640, save_period=5, patience=10):
        """Train the YOLOv8 model."""
        print(f"Initializing YOLO model with '{self.model_name}'...")
        self.model = YOLO(self.model_name)
        
        print(f"Starting training with dataset: {dataset_yaml}")
        results = self.model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=self.project_dir,
            name="train_results",
            save_period=save_period,
            patience=patience,
            verbose=True,
            device=0 if torch.cuda.is_available() else 'cpu' 
        )
        
        self.trained_model_path = os.path.join(self.project_dir, "train_results", "weights", "best.pt")
        print(f"Model trained successfully. Best weights saved to {self.trained_model_path}")
        return self.trained_model_path
    
    def load_model(self, model_path):
        """Load a trained YOLOv8 model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = YOLO(model_path)
        print(f"Model loaded from {model_path}")
        return self.model.names
    
    def get_color_mapping(self, class_names):
        """Generate color mapping for visualization based on team affiliation."""
        colors = {}
        team_mapping = self.load_team_mapping()
        
        for class_id, name in class_names.items():
            if name == "ball":
                colors[class_id] = self.team_colors["ball"]
            else:
                team = team_mapping.get(name, "").lower() 
                
                if not team: # Fallback if not explicitly in mapping
                    if "barcelona" in name.lower():
                        team = "barcelona"
                    elif any(term in name.lower() for term in ["madrid", "real"]):
                        team = "real_madrid"
                
                colors[class_id] = self.team_colors.get(team, self.team_colors["unknown"])
        
        return colors
    
    def predict_image(self, image_path, conf=0.3, save_results=True, visualize=True):
        """Run prediction on a single image."""
        if self.model is None:
            raise ValueError("Model not loaded. Use load_model() first.")
            
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        print(f"Running prediction on image: {image_path}")
        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=save_results,
            project=self.project_dir,
            name="image_results",
            exist_ok=True 
        )
        
        if visualize:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path} for visualization.")
                return results
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            color_mapping = self.get_color_mapping(self.model.names)
            
            for r in results:
                if r.boxes.xyxy.numel() == 0: 
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = box.astype(int)
                    class_id = int(cls)
                    label = f"{r.names[class_id]}: {conf:.2f}"
                    color = color_mapping.get(class_id, (255, 255, 255))
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(f"Predictions for {os.path.basename(image_path)}")
            plt.axis('off')
            plt.show()
        
        return results
    
    def process_video(self, video_path, conf=0.3, save_video=True, track=True):
        """Process video for player and ball tracking."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        color_mapping = self.get_color_mapping(self.model.names)
        tracker = sv.ByteTrack() if track else None
        box_annotator = sv.BoxAnnotator() if track else None 
        
        output_path = None
        out = None
        if save_video:
            output_dir = os.path.join(self.project_dir, "video_results")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"processed_{os.path.basename(video_path)}")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        print(f"Starting video processing for {video_path}...")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model.predict(source=frame, conf=conf, verbose=False)[0]
            
            annotated_frame = frame.copy() 
            
            if track and len(results.boxes.xyxy) > 0:
                detections = sv.Detections(
                    xyxy=results.boxes.xyxy.cpu().numpy(),
                    confidence=results.boxes.conf.cpu().numpy(),
                    class_id=results.boxes.cls.cpu().numpy().astype(int)
                )
                
                detections = tracker.update(detections=detections)
                
                labels = []
                colors_for_frame = []
                for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id):
                    class_name = self.model.names[class_id]
                    labels.append(f"#{tracker_id} {class_name}: {confidence:.2f}")
                    colors_for_frame.append(color_mapping.get(class_id, (255, 255, 255)))

                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame, 
                    detections=detections, 
                    labels=labels,
                    colors=colors_for_frame
                )
            else: 
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                confidences = results.boxes.conf.cpu().numpy()
                
                for box, cls_id, conf_val in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = box.astype(int)
                    label = f"{self.model.names[cls_id]}: {conf_val:.2f}"
                    color = color_mapping.get(cls_id, (255, 255, 255))
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if out is not None:
                out.write(annotated_frame)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")
        
        cap.release()
        if out is not None:
            out.release()
            
        print(f"Video processing complete. Output saved to {output_path}" if save_video else "Video processing complete")
        return output_path if save_video else None

    def generate_labels_from_filenames(self, images_dir, output_dir=None, player_names=None, default_box_size=(0.1, 0.2)):
        """
        Generate YOLO format labels from image filenames. 
        This is a utility function for initial dummy labels or if your dataset is named in a way that suggests classes.
        For proper training, human-annotated labels are highly recommended.
        """
        if player_names is None:
            raise ValueError("Player names list is required")
            
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(images_dir), "labels")
        
        os.makedirs(output_dir, exist_ok=True)
        
        class_mapping = {name.lower(): i for i, name in enumerate(player_names)}
        if "ball" not in class_mapping: 
            class_mapping["ball"] = len(player_names)
            
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        label_count = 0
        
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            labels = []
            
            filename_lower = base_name.lower().replace("_", " ").replace("-", " ")
            
            detected_players = []
            for player_name, class_id in class_mapping.items():
                player_name_norm = player_name.lower().replace("_", " ").replace("-", " ")
                if player_name_norm in filename_lower or any(
                    part in filename_lower for part in player_name_norm.split() if len(part) > 3
                ):
                    detected_players.append((player_name, class_id))
            
            default_width, default_height = default_box_size
            
            if detected_players:
                player_count = len(detected_players)
                for idx, (player_name, class_id) in enumerate(detected_players):
                    x_center = (idx + 1) / (player_count + 1)
                    y_center = 0.5
                    label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {default_width:.6f} {default_height:.6f}"
                    labels.append(label_line)
            
            if "ball" in filename_lower:
                ball_class_id = class_mapping.get("ball")
                if ball_class_id is not None:
                    ball_label = f"{ball_class_id} 0.500000 0.500000 0.050000 0.050000"
                    labels.append(ball_label)
            
            if labels:
                label_file = os.path.join(output_dir, f"{base_name}.txt")
                with open(label_file, 'w') as f:
                    f.write('\n'.join(labels))
                label_count += 1
        
        print(f"Generated {label_count} label files in {output_dir}")
        return label_count
    
    def create_team_dataset(self, barcelona_players, real_madrid_players, dataset_root_dir):
        """
        Creates dataset configuration (YAML file and team mapping) for training,
        assuming images and labels are already organized within dataset_root_dir.
        
        Args:
            barcelona_players (list): List of Barcelona player names.
            real_madrid_players (list): List of Real Madrid player names.
            dataset_root_dir (str): The root directory of the pre-organized dataset.
        
        Returns:
            dict: Contains team_mapping, yaml_path, and all_players list.
        """
        all_players = []
        team_mapping = {}
        
        for player in barcelona_players:
            all_players.append(player)
            team_mapping[player] = "barcelona"
            
        for player in real_madrid_players:
            all_players.append(player)
            team_mapping[player] = "real_madrid"
            
        all_players.append("ball")
        team_mapping["ball"] = "equipment" # General category for ball
        
        self.save_team_mapping(team_mapping)
        
        yaml_path, class_dict = self.create_dataset_yaml(
            dataset_path=dataset_root_dir,
            train_dir_name="train", 
            val_dir_name="val",     
            player_names=all_players,
            include_ball=True,
            team_mapping=team_mapping
        )
        
        return {
            "team_mapping": team_mapping,
            "yaml_path": yaml_path,
            "all_players": all_players
        }
    
    def train_model(self, dataset_dir, epochs=50, batch_size=8, imgsz=640):
        """
        Runs the complete training pipeline from a pre-organized dataset directory to a trained model.
        
        Args:
            dataset_dir (str): Path to the root of the dataset structured as:
                                dataset_dir/
                                ├── train/
                                │   ├── images/
                                │   └── labels/
                                └── val/
                                    ├── images/
                                    └── labels/
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            imgsz (int): Image size for training.
            
        Returns:
            str: Path to the best trained model weights.
        """
        print("=== Starting Training Pipeline ===")
        print(f"Using dataset from: {dataset_dir}")
        
        # Validate dataset directory structure
        required_dirs = [
            os.path.join(dataset_dir, "train", "images"),
            os.path.join(dataset_dir, "train", "labels"),
            os.path.join(dataset_dir, "val", "images"),
            os.path.join(dataset_dir, "val", "labels"),
        ]
        
        for r_dir in required_dirs:
            if not os.path.exists(r_dir):
                raise FileNotFoundError(
                    f"Required directory not found: {r_dir}. "
                    "Please ensure your dataset is structured correctly."
                )

        # Define all players for classification. These *must* match your annotation classes.
        barcelona_players = [
            "Gavi", "Szczęsny", "Koundé", "Iñigo Martínez", "Cubarsí",
            "Alejandro Balde", "Marc Casadó", "Pedri", "Raphinha",
            "Lamine Yamal", "Lewandowski"
        ]
        
        real_madrid_players = [
            "Thibaut Courtois", "Ferland Mendy", "Rüdiger", "Aurélien Tchouaméni",
            "Lucas Vázquez", "Federico Valverde", "Eduardo Camavinga",
            "Bellingham", "Vinicius Junior", "Mbappé"
        ]

        # Create dataset configuration YAML
        dataset_info = self.create_team_dataset(
            barcelona_players=barcelona_players,
            real_madrid_players=real_madrid_players,
            dataset_root_dir=dataset_dir 
        )

        # Train model using the generated YAML
        trained_model_path = self.train(
            dataset_yaml=dataset_info["yaml_path"],
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            save_period=5,
            patience=10
        )

        print(f"\n=== Training complete! Model saved to: {trained_model_path} ===")
        return trained_model_path
    
    def test_model(self, test_data=None, conf_threshold=0.3, display_results=True):
        """Test the trained model on images or videos."""
        if self.model is None:
            raise ValueError("Model not loaded.")
            
        if test_data is None:
            test_data = input("Enter path to test image/video/directory: ")
            
        if not os.path.exists(test_data):
            raise FileNotFoundError(f"Test data not found at {test_data}")

        results_outputs = [] 

        if os.path.isdir(test_data):
            print(f"Testing on directory: {test_data}")
            image_files = [f for f in os.listdir(test_data) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                print(f"Found {len(image_files)} images to process.")
            for img_file in image_files:
                img_path = os.path.join(test_data, img_file)
                self.predict_image(
                    image_path=img_path,
                    conf=conf_threshold,
                    save_results=True,
                    visualize=display_results
                )
                results_outputs.append(os.path.join(self.project_dir, "image_results", os.path.basename(img_path)))
                
            video_files = [f for f in os.listdir(test_data) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if video_files:
                print(f"Found {len(video_files)} videos to process.")
            for vid_file in video_files:
                vid_path = os.path.join(test_data, vid_file)
                output_path = self.process_video(
                    video_path=vid_path,
                    conf=conf_threshold,
                    save_video=True,
                    track=True
                )
                if output_path:
                    results_outputs.append(output_path)
                
        elif test_data.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Testing on video: {test_data}")
            output_path = self.process_video(
                video_path=test_data,
                conf=conf_threshold,
                save_video=True,
                track=True
            )
            if output_path:
                results_outputs.append(output_path)
            
        else: # Assume single image
            print(f"Testing on image: {test_data}")
            self.predict_image(
                image_path=test_data,
                conf=conf_threshold,
                save_results=True,
                visualize=display_results
            )
            results_outputs.append(os.path.join(self.project_dir, "image_results", os.path.basename(test_data)))
            
        print("\n--- Test complete ---")
        if results_outputs:
            print("Results saved/processed for:")
            for res_path in results_outputs:
                print(f"- {res_path}")
        return results_outputs
        
    def test_video_with_boxes(self, video_path, output_path=None, conf=0.3, display=False, 
                                track=True, show_progress=True):
        """
        Process video with team-colored bounding boxes and player identification.
        
        Args:
            video_path (str): Path to input video.
            output_path (str, optional): Path to save output video. If None, video is not saved.
            conf (float): Confidence threshold (0-1) for detections.
            display (bool): Whether to show the video preview using matplotlib in real-time.
            track (bool): Whether to use ByteTrack for object tracking across frames.
            show_progress (bool): Whether to show processing progress in the console.
            
        Returns:
            str: Path to output video if saved, None otherwise.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Use load_model() first.")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        team_mapping = self.load_team_mapping()
        
        writer = None
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        tracker = ByteTrack() if track else None
        
        fig, ax = None, None
        if display:
            plt.ion()  
            fig, ax = plt.subplots(figsize=(12, 8))
            img_display = ax.imshow(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))
            ax.set_title(f"Video Stream: {os.path.basename(video_path)}")
            plt.tight_layout()
            plt.show(block=False)
        
        import time
        
        try:
            frame_count = 0
            start_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run prediction
                results = self.model(frame, conf=conf, verbose=False)[0]
                
                annotated_frame = frame.copy()
                
                if len(results.boxes) > 0:
                    detections = Detections.from_ultralytics(results)
                    
                    if tracker:
                        detections = tracker.update(detections=detections) 
                    
                    for i in range(len(detections)):
                        xyxy = detections.xyxy[i]
                        confidence = detections.confidence[i]
                        class_id = detections.class_id[i]
                        tracker_id = detections.tracker_id[i] if tracker and detections.tracker_id is not None else -1
                        
                        class_name = self.model.names[class_id]
                        
                        team = team_mapping.get(class_name, "").lower() 
                        
                        if not team:
                            if "barcelona" in class_name.lower():
                                team = "barcelona"
                            elif any(term in class_name.lower() for term in ["madrid", "real"]):
                                team = "real_madrid"
                            elif "ball" in class_name.lower():
                                team = "ball"
                        
                        color = self.team_colors.get(team, self.team_colors["unknown"])
                        
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1),
                            (x2, y2),
                            color=color,
                            thickness=2
                        )
                        
                        label = f"{class_name}: {confidence:.2f}"
                        if tracker_id != -1: 
                            label = f"#{tracker_id} {label}"
                        
                        label_size, baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        y1_label_pos = max(y1, label_size[1] + 10) 
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1_label_pos - label_size[1] - 10),
                            (x1 + label_size[0], y1_label_pos),
                            color,
                            cv2.FILLED
                        )
                        
                        text_color = (0, 0, 0) if sum(color) > 382 else (255, 255, 255) 
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1_label_pos - 7),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            text_color,
                            2
                        )
                
                frame_count += 1
                elapsed = time.time() - start_time
                processing_fps = frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(
                    annotated_frame,
                    f"FPS: {processing_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0), 
                    2
                )
                
                if writer:
                    writer.write(annotated_frame)
                
                if display:
                    img_display.set_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                    ax.set_title(f"Video Stream: {os.path.basename(video_path)} | Frame {frame_count}/{total_frames} | FPS: {processing_fps:.1f}")
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                
                if show_progress and frame_count % 30 == 0: 
                    print(f"Processing: {frame_count}/{total_frames} frames ({processing_fps:.1f} FPS)", end='\r')
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                plt.ioff()
                plt.close(fig)
            
            if show_progress:
                print(f"\nProcessed {frame_count} frames in {elapsed:.1f} seconds (Average {frame_count/elapsed:.1f} FPS)")
                if output_path:
                    print(f"Output video saved to: {output_path}")
        
        return output_path if writer else None

if __name__ == "__main__":
    # Check for CUDA availability
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA enabled - Using GPU acceleration")
    else:
        print("CUDA not available - Using CPU")

    # --- Setup for Training ---
    # IMPORTANT: Ensure your dataset is organized as follows before running:
    # my_football_dataset/
    # ├── train/
    # │   ├── images/
    # │   │   ├── image1.jpg
    # │   │   └── ...
    # │   └── labels/
    # │       ├── image1.txt
    # │       └── ...
    # └── val/
    #     ├── images/
    #     │   ├── image_val1.jpg
    #     │   └── ...
    #     └── labels/
    #         ├── image_val1.txt
    #         └── ...

    # Set the root directory where your 'train' and 'val' folders are located.
    DATASET_ROOT = "C:/Users/ADMIN/Desktop/ahmad/data_train/dataset_football" # <--- ADJUST THIS PATH

    # Initialize the tracker for training. It will download yolov8n.pt if not found locally.
    tracker_for_training = PlayerBallTracker(model_name="yolov8n.pt", project_dir="my_training_run")

    try:
        # Start the training pipeline
        trained_model_path = tracker_for_training.train_model(
            dataset_dir=DATASET_ROOT,
            epochs=100,      # Number of training epochs
            batch_size=16,   # Adjust based on your GPU memory
            imgsz=640        # Image size for training
        )
        print(f"Training complete. Best model saved at: {trained_model_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please verify your DATASET_ROOT path and ensure the dataset structure is correct.")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
    
    print("\n" + "="*50 + "\n")

    # --- Setup for Inference (using a trained model) ---
    # This section demonstrates how to use the trained model for video processing.
    
    # If training was successful, use the path to the newly trained model.
    # Otherwise, you can manually set a path to a pre-existing trained model.
    INFERENCE_MODEL_PATH = trained_model_path if 'trained_model_path' in locals() and trained_model_path else None
    
    # Fallback if no model path is set from training (e.g., if you only want to run inference)
    if not INFERENCE_MODEL_PATH or not os.path.exists(INFERENCE_MODEL_PATH):
        print("Warning: No newly trained model found or specified. Attempting to use a default path.")
        # Make sure this path points to a valid .pt model file you want to use for inference.
        INFERENCE_MODEL_PATH = "C:/Users/ADMIN/Desktop/ahmad/data_train/my_training_run/train_results/weights/best.pt" 

    if INFERENCE_MODEL_PATH and os.path.exists(INFERENCE_MODEL_PATH):
        print(f"Loading model for inference: {INFERENCE_MODEL_PATH}")
        tracker_for_inference = PlayerBallTracker(model_path=INFERENCE_MODEL_PATH, project_dir="my_inference_results")

        # Define the video file to process
        VIDEO_TO_PROCESS = "cropped_video.mp4" # <--- ADJUST THIS PATH to your input video
        OUTPUT_VIDEO_PATH = "output_tracked_final.mp4" # Name for the output video

        if os.path.exists(VIDEO_TO_PROCESS):
            print(f"Starting video processing for {VIDEO_TO_PROCESS}...")
            tracker_for_inference.test_video_with_boxes(
                video_path=VIDEO_TO_PROCESS,
                output_path=OUTPUT_VIDEO_PATH, 
                conf=0.3,       # Confidence threshold for detections
                display=True,   # Set to True to visualize while processing (requires matplotlib)
                track=True      # Enable tracking across frames using ByteTrack
            )
        else:
            print(f"Error: Video file not found at {VIDEO_TO_PROCESS}. Skipping video processing.")
    else:
        print(f"Error: No valid trained model found at {INFERENCE_MODEL_PATH}. Cannot perform inference.")