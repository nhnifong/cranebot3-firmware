from ursina import (
    Ursina,
    Entity,
    Button,
    EditorCamera,
    color,
)

app = Ursina()

# Create the cube
cube = Entity(model='cube', color=color.orange, scale=(1, 1, 1), position=(2,0,0))

# Create the button
def on_button_click():
    print("Foo button clicked!")
    cube.color = color.random_color()


foo_button = Button(text='Foo', color=color.azure, position=(-.7, 0), scale=(.3, .1), on_click=on_button_click)


# ground = Entity(model='quad', scale_x=10, collider='box', color=color.black)
# player = PlatformerController2d(y=1, z=.01, scale_y=1, max_jumps=2)

# def update():
#     player.y = player.y + time.dt * 0.5

EditorCamera()

app.run()