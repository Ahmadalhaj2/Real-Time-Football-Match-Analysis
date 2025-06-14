import sys
sys.path.append('../')

from utils import get_center_of_bbox ,measure_distance


class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
        
    def calculate_distance_between_ball_player(self,bbox_player,bbox_ball):
        center_of_ball = get_center_of_bbox(bbox_ball)
        center_of_player = get_center_of_bbox(bbox_player)
        return measure_distance(center_of_player,center_of_ball)  
        
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)
        
        minimum_distance = 99999
        assigned_player = -1
        
        for player_id,player in players.items():
            player_bbox = player['bbox']
            
            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)   
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position) 
            distance = min(distance_left,distance_right)   
            
            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id
                    
        return assigned_player
    
    def calculate_numberof_touches_for_player(self,number_of_players,ball_box):
        touches_total =0
        
        for _,player in number_of_players.items():   
            player_bbox = player['bbox']
            
            x_distance = player_bbox[0] - player_bbox[2]
            
            print("distance for bbox for player = ", x_distance)
            # y_distance = player_bbox[2] - player_bbox[3]
            distance_for_touch = self.calculate_distance_between_ball_player(player_bbox,ball_box)

            if abs(distance_for_touch) < (int(abs(x_distance)/2)):
                touches_total +=1
            
                
        return touches_total
    
        
 
        