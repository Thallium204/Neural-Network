extends Control

onready var NN = $NeuralNetwork
onready var nodeFont = load("res://nodeFont.tres")

var DECPLACES = 2

var nodePositions = {}

var radius = 40

func setNodePositions():
	nodePositions = NN.nodes.duplicate(true)
	var numLayers = NN.size
	# DRAW NODES
	for layerID in numLayers:
		var x_pos = layerID * rect_size.x/(numLayers-1)
		var numNodes = nodePositions[layerID].size()
		for nodeID in numNodes:
			var y_pos = nodeID * rect_size.y/(numNodes-1)
			var nodePos = Vector2(x_pos,y_pos)
			nodePositions[layerID][nodeID] = [nodePos]

func _ready():
	setNodePositions()

func _draw():
	
	var numLayers = NN.size
	
	# DRAW WEIGHTS
	for layerIDPair in NN.weights:
		var weightMatrix = NN.weights[layerIDPair]
		for fromNodeID in range(weightMatrix.size()):
			for toNodeID in range(weightMatrix[fromNodeID].size()):
				var valueFloat = weightMatrix[fromNodeID][toNodeID]
				var weightColor = Color(-min(valueFloat,0),max(0,valueFloat),0)
				draw_line(
					nodePositions[layerIDPair[0]][toNodeID][0],
					nodePositions[layerIDPair[1]][fromNodeID][0],
					weightColor,5)
	
	# DRAW NODE STRING
	for layerID in numLayers:
		var numNodes = NN.nodes[layerID].size()
		for nodeID in numNodes:
			# Get string of float
			var valueFloat = NN.nodes[layerID][nodeID][0]
			var valueString = str(stepify(valueFloat,1/pow(10,DECPLACES)))
			var nodeStrSize = nodeFont.get_string_size(valueString)
			var nodePos = nodePositions[layerID][nodeID][0]
			draw_circle(nodePos,radius,Color.white)
			var nodeStrPos = nodePos + Vector2(-nodeStrSize.x/2,nodeStrSize.y/3)
			var nodeColor = Color(0,0,0)
			draw_string(nodeFont,nodeStrPos,valueString,nodeColor)

func _process(_delta):
	update()

func _on_Network_item_rect_changed():
	setNodePositions()


