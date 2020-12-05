extends Control

var radius = 30

func _draw():
	draw_circle(rect_position,radius,Color.white)

func _process(delta):
	radius += 1
	update()
