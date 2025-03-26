from ursina import *
from ursina.shaders import (
    lit_with_shadows_shader,
    unlit_shader,
)

class SplineMovingEntity(Entity):
    """
    An entity that moves it's position along a spline function
    """
    def __init__(self, ui, spline_func, model, **kwargs):
        super().__init__(
            model=model,
            **kwargs
        )
        self.is_gantry = (model == 'gantry')
        self.ui = ui
        self.spline_func = spline_func
        self.time_domain = (333,444)

    def setParams(self, func, domain):
        self.spline_func = func
        self.time_domain = domain

    def update(self):
        if self.spline_func is not None:
            t = (time.time() - self.time_domain[0]) / (self.time_domain[1] - self.time_domain[0])
            self.position = self.spline_func(t)
            if self.is_gantry:
                self.redraw_wires()

    def redraw_wires(self):
        # update the lines between the gantry and the other things
        for anchor_num in range(self.ui.n_anchors):
            self.ui.lines[anchor_num].model = draw_line(self.ui.anchors[anchor_num].position, self.position)
        self.ui.vert_line.model = draw_line(self.ui.gripper.position, self.position)


class Gripper(SplineMovingEntity):
    def __init__(self, ui, spline_func, **kwargs):
        super().__init__(
            ui=ui,
            spline_func=spline_func,
            collider='box', # for mouse interaction
            model='gripper_body',
            scale=0.001,
            color=(0.9, 0.9, 0.9, 1.0),
            **kwargs
        )
        self.label_offset = (0.00, 0.04)
        self.label = Text(
            color=(0.1,0.1,0.1,1.0),
            text=f"Gripper\nNot Detected",
            scale=0.6,
        )
        self.left_finger = Entity(
            parent=self,
            model='gripper_finger',
            position=(-25,41,-10.5),
            color=(0.0, 0.2, 0.0, 1.0),
        )
        self.right_finger = Entity(
            parent=self,
            model='gripper_finger',
            position=(25,41,-10.5),
            scale=(-1,1,1),
            color=(0.0, 0.2, 0.0, 1.0),
        )


    def setStatus(self, status):
        self.label.text = f"Gripper\n{status}"

    def update(self):
        super().update()
        self.label.position = world_position_to_screen_position(self.position) + self.label_offset
        # if time.time() > self.last_ob_render+0.5:
        #     self.ui.render_gripper_ob()
        #     self.last_ob_render = time.time()

    def on_mouse_enter(self):
        self.color = anchor_color_selected

    def on_mouse_exit(self):
        self.color = anchor_color

    def on_click(self):
        self.wp = WindowPanel(
        title=f"Gripper Controls",
        content=(
            Button(text='Open', color=color.gold, text_color=color.black),
            Button(text='Show video feed', color=color.gold, text_color=color.black),
            Button(text='Autofocus', color=color.gold, text_color=color.black),
            Button(text='Stop Spool Motor', color=color.gold, text_color=color.black),
            Button(text='Reel in 20cm', color=color.orange, text_color=color.black, on_click=self.reel_manual),
            Button(text='Reel out 20cm', color=color.orange, text_color=color.black, on_click=self.reel_manual),
            Button(text='Sleep', color=color.gold, text_color=color.black),
            ),
        popup=True
        )

    def reel_manual(self, delta_meters):
        self.wp.enabled = False

    def setAppearanceClosed(self, closed):
        if closed:
            self.left_finger.rotation = (0,0,0)
            self.right_finger.rotation = (0,0,0)
        else:
            self.left_finger.rotation = (0,0,60)
            self.right_finger.rotation = (0,0,-60)


anchor_color = (0.8, 0.8, 0.8, 1.0)
anchor_color_selected = (0.9, 0.9, 1.0, 1.0)
class Anchor(Entity):
    def __init__(self, num, to_ob_q, position, rotation=(0,0,0)):
        super().__init__(
            position=position,
            rotation=rotation,
            model='anchor',
            color=anchor_color,
            scale=1,
            shader=lit_with_shadows_shader,
            collider='box'
        )
        self.num = num
        self.label_offset = (0.00, 0.04)
        self.to_ob_q = to_ob_q
        self.pose = np.array((rotation, position))

        self.label = Text(
            color=(0.1,0.1,0.1,1.0),
            text=f"Anchor {self.num}\nNot Detected",
            scale=0.5,
        )

        self.empty = Entity(
            scale=1,
            # rotation=(-35,-90,180),
            rotation=to_ursina_rotation(compose_poses([
                model_constants.anchor_camera,
                (np.array([pi/2,0,0], dtype=float), np.array([0,0,0], dtype=float))
                ])[0]),
            parent=self)

        self.camview = Entity(
            model='quad',
            scale=(2, 2/1.777777),
            position=(0,0,2),
            texture='cap_38.jpg',
            shader=unlit_shader,
            parent=self.empty,
            enabled=False)

        # flag for doing something once
        self.hasSetImage = False

    def setStatus(self, status):
        self.label.text = f"Anchor {self.num}\n{status}"

    def update(self):
        self.label.position = world_position_to_screen_position(self.position) + self.label_offset

    def on_mouse_enter(self):
        self.color = anchor_color_selected

    def on_mouse_exit(self):
        self.color = anchor_color

    def on_click(self):
        self.wp = WindowPanel(
        title=f"Anchor {self.num}",
        content=(
            Button(text='Show video feed', color=color.gold, text_color=color.black),
            Button(text='Autofocus', color=color.gold, text_color=color.black),
            Button(text='Stop Spool Motor', color=color.gold, text_color=color.black),
            Button(text='Reel in 20cm', color=color.orange, text_color=color.black,
                on_click=partial(self.reel_manual, -0.05)),
            Button(text='Reel out 20cm', color=color.orange, text_color=color.black,
                on_click=partial(self.reel_manual, 0.05)),
            Button(text='Sleep', color=color.gold, text_color=color.black),
            ),
        popup=True
        )

    def reel_manual(self, delta_meters):
        self.wp.enabled = False
        self.to_ob_q.put({'jog_spool':{'anchor':self.num, 'rel':delta_meters}})        


class Floor(Entity):
    def __init__(self, **kwargs):
        super().__init__(
            collider='box',
            **kwargs
        )
        self.alt = 0.3 # altitude of blue circle in meters
        self.target = Entity(
            model='quad',
            position=(0, 0, 0),
            rotation=(90,0,0),
            color=color.white,
            scale=(0.5, 0.5, 0.5),
            texture='green_target.png',
        )
        self.pipe = Entity(
            model=Pipe(
                path=[(0,0,0), (0,self.alt*2,0)], # have to multiply by 2 because it inherited the scale of it's parent
                thicknesses=(0.01, 0.01),
                cap_ends=True),
            rotation=(-90,0,0),
            parent=self.target,
            color=color.white,
        )
        self.circle = Entity(
            model='quad',
            position=(0, 0, -self.alt*2),
            color=color.white,
            scale=(0.2, 0.2, 0.2),
            parent=self.target,
            texture='blue_circle.png',
        )

    def on_click(self,):
        print(mouse.world_point)
        # send message to position estimator with desired future position

    def update(self,):
        if mouse.hovered_entity == self:
            # Get the intersection point in world coordinates
            self.target.position = mouse.world_point

class GoalPoint(Entity):
    def __init__(self, **kwargs):
        super().__init__(
            position=(0,0,0),
            rotation=(-90,0,0),
            model='map_marker',
            color=color.azure,
            scale=0.075,
            shader=lit_with_shadows_shader,
            enabled=True,
            **kwargs
        )
        self.atime = time.time()+10
        self.fmt = 'ETA {val:.2f}'
        self.label_offset = (-0.02, -0.03)
        self.label = Text(
            color=(0.1,0.1,0.1,1.0),
            text=f"",
            scale=0.5,
        )

    def update(self):
        self.label.position = world_position_to_screen_position(self.position) + self.label_offset
        self.label.text = self.fmt.format(val=(time.time()-self.atime))

class EntityPool:
    def __init__(self, max_items, new_entity_fn):
        self.entities = []
        self.max_items = max_items
        self.next = 0
        self.new_entity_fn = new_entity_fn

    def replace(self, item_list, update_entity_fn):
        """show up to max_items from items_list and nothing more
        update_entity_fn(a, b) will be called for each item in item_list, where a is the item and b is the entity to be updated
        disable anything else in the pool that was not updated by this call
        """
        i = 0
        while i < min(len(item_list), self.max_items):
            # show an item with this entity
            if len(self.entities) <= i:
                self.entities.append(self.new_entity_fn())
            update_entity_fn(item_list[i], self.entities[i])
            self.entities[i].enabled = True
            i += 1
        # disable whatever remains
        while i < len(self.entities):
            self.entities[i].enabled = False
            i += 1
        self.next = 0

    def add(self, update_entity_fn):
        """show a new item, reusing the oldest one if needed"""
        if len(self.entities) < self.max_items:
            self.entities.append(self.new_entity_fn())
            update_entity_fn(self.entities[-1])
            self.entities[-1].enabled = True
        else:
            update_entity_fn(self.entities[self.next])
            self.entities[-1].enabled = True
            self.next = (self.next + 1) % self.max_items

class ThinSliderLog2(ThinSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # feedback for the errors of the last minimizer step
        self.errorbar = Entity(
            parent=self.bg,
            model=Quad(
                radius=0.01,
                segments=3),
            origin_x=-0.5,
            scale=(0.5, 0.035),
            color=color.red,
            z=0.2,
        )

        self._update_text()

    def setErrorBar(self, value):
        """value expected in range of 0 to 1"""
        self.errorbar.scale = (value/2, 0.035)

    def finalValue(self):
        return 2**(self.value-5)

    def _update_text(self):
        self.knob.text_entity.text = str(round(self.finalValue(), 3))