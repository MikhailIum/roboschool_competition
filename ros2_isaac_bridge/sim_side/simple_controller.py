import random
import time
from enum import Enum

class RobotState(Enum):
    SEEKING = "searching"      # Поиск объекта
    APPROACHING = "approaching" # Приближение к объекту
    WALL_FOLLOWING = "wall_following"  # Движение вдоль стены

class Robot:
    def __init__(self):
        self.state = RobotState.SEEKING
        self.wall_direction = 1  # 1 = влево, -1 = вправо
        self.last_wall_time = 0
        self.search_time = 0
        
    def get_camera_frame(self):
        # Получение кадра с камеры
        pass
    
    def detect_object(self, frame):
        # Запуск нейросети на кадре
        # Возвращает: (found, x_center, y_center, distance)
        pass
    
    def move_forward(self, speed):
        # Движение вперед
        pass
    
    def turn(self, angle):
        # Поворот на угол (градусы)
        pass
    
    def get_wall_distance(self):
        # Получение расстояния до стены (ультразвук/лидар)
        pass
    
    def main_loop(self):
        while True:
            frame = self.get_camera_frame()
            object_detected, x, y, distance = self.detect_object(frame)
            
            if object_detected and distance < 300:  # Объект в зоне видимости
                self.state = RobotState.APPROACHING
                self.approach_object(x, distance)
            else:
                if self.state == RobotState.APPROACHING:
                    # Потеряли объект - переключаемся в поиск
                    self.state = RobotState.SEEKING
                    self.search_time = time.time()
                
                self.wall_following_search()
            
            time.sleep(0.05)  # 20 FPS
    
    def approach_object(self, x_center, distance):
        """Приближение к обнаруженному объекту"""
        frame_center = 320  # Допустим, ширина кадра 640px
        
        # Коррекция направления
        error = x_center - frame_center
        if abs(error) > 50:  # Объект не по центру
            turn_angle = error * 0.3  # Пропорциональный поворот
            self.turn(turn_angle)
        else:
            # Объект по центру - едем прямо
            speed = min(50, distance / 2)  # Замедляемся при приближении
            self.move_forward(speed)
        
        # Если очень близко - остановка
        if distance < 50:
            self.stop_and_interact()
    
    def wall_following_search(self):
        """Поиск объекта методом от стены к стене"""
        wall_distance = self.get_wall_distance()
        
        # Пороги расстояния до стены
        WALL_CLOSE = 30    # Слишком близко к стене
        WALL_FAR = 100     # Слишком далеко от стены
        
        if wall_distance < WALL_CLOSE:
            # Слишком близко к стене - отворачиваем
            self.turn(90 * self.wall_direction)
            self.move_forward(30)
            
        elif wall_distance > WALL_FAR:
            # Слишком далеко от стены - ищем следующую стену
            self.search_for_wall()
            
        else:
            # Идеальная дистанция - двигаемся вдоль стены
            self.move_forward(40)
            
            # Периодически меняем направление (случайный фактор)
            if random.random() < 0.05:  # 5% шанс каждую итерацию
                self.change_search_direction()
    
    def search_for_wall(self):
        """Поиск стены при потере контакта"""
        # Поворачиваемся на месте
        self.turn(15 * self.wall_direction)
        self.move_forward(20)
        
        # Если не нашли стену за 2 секунды - меняем направление
        if time.time() - self.last_wall_time > 2:
            self.wall_direction *= -1
            self.last_wall_time = time.time()
    
    def change_search_direction(self):
        """Случайная смена направления движения"""
        # 50% шанс изменить направление вдоль стены
        if random.random() < 0.5:
            self.wall_direction *= -1
        else:
            # Случайный поворот
            random_angle = random.randint(-45, 45)
            self.turn(random_angle)
            self.move_forward(30)
    
    def stop_and_interact(self):
        """Остановка при достижении объекта"""
        self.move_forward(0)
        print("Объект достигнут!")
        # Здесь можно добавить действие: взять объект, сигнал и т.д.
        time.sleep(2)
        self.state = RobotState.SEEKING