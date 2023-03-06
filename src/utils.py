from collections import defaultdict


class Human():

    def __init__(self, frame, track_id, coord) -> None:
        self.frame = int(frame) - 1
        self.track_id = int(track_id)
        self.coord = list(map(float, coord))
        self.area = self.get_area()

    def get_area(self):
        return (self.area[2] - self.area[0]) * (self.area[3] - self.area[1]) 
    def xywh_to_xyxy(self):
        x, y, w, h = self.coord
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
        return (xmin, ymin, xmax, ymax)


def get_object_frame(file):
    with open(file, "rt") as f:
        lines = f.readlines()
        
    lines = list(map(lambda x: x.strip().split(",")[:6], lines))

    humans = []
    for line in lines:
        humans.append(Human(line[0], line[1], line[2:]))

    group_frame = defaultdict(list)
    for human in humans:
        group_frame[human.frame].append(human)
    return group_frame