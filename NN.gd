extends Node

var rng = RandomNumberGenerator.new()

var nodes = {}
var weights = {}
var activations = {}
var size = 0

var dataset = []
var trainSetIDs = []
var testSetIDs = []

var costFunction = 0
var costHistory = []

enum AType{
	LINEAR,
	SOFTMAX,
	MAX
}

enum CostFunc{
	ABSOLUTE,
	SQUARED
}

func _ready():
	var netID = initNetwork([
		{"size":3,"AType":AType.LINEAR}, # INPUT
		{"size":4,"AType":AType.LINEAR},
		{"size":2,"AType":AType.SOFTMAX}], # OUTPUT
	generateExampleDataset(250),
	CostFunc.ABSOLUTE)
	generateTrainAndTest(0.60)
	test()

func getStatus(context="general"):
	print()
	print("["+context+"]")
	print("nodes: ",nodes)
	print("weights: ",weights)
	print("activations: ",activations)
	#print("trainSetIDs: ",trainSetIDs)
	#print("testSetIDs: ",testSetIDs)
	print("costFunction: ",costFunction)
	print("costHistory: ",costHistory)
	#print("dataset: ",dataset)

func generateExampleDataset(setSize:int):
	var ex_dataset = []
	var inputs = []
	while ex_dataset.size() < setSize:
		rng.randomize()
		var example = [
			[rng.randi_range(0,10)],
			[rng.randi_range(0,10)],
			[rng.randi_range(0,10)]
		]
		var isEven = (example[0][0]+example[1][0]+example[2][0])%2
		if not example in inputs:
			inputs.append(example)
			ex_dataset.append(
				{"input":example,
				"output":[[isEven],[1-isEven]]}
			)
	
	return ex_dataset

func generateTrainAndTest(percTrain):
	var sampleSize = dataset.size()
	var trainSize = int(sampleSize * percTrain)
	var datasetIDs = range(dataset.size())
	datasetIDs.shuffle()
	trainSetIDs = datasetIDs.slice(0,trainSize-1)
	testSetIDs = datasetIDs.slice(trainSize,sampleSize-1)
	getStatus("train and test")

func initNetwork(networkConfig:Array,givenDataset:Array,givenCostFunc:int):
	#print(dataset)
	
	nodes = {}
	weights = {}
	activations = {}
	size = networkConfig.size()
	dataset = givenDataset
	costFunction = givenCostFunc
	
	# Ensure network dimensions fit dataset
	networkConfig[0].size = dataset[0].input.size()
	networkConfig[-1].size = dataset[0].output.size()
	
	for layerID in range(networkConfig.size()):
		var layer = networkConfig[layerID]
		nodes[layerID] = Matrix(layer.size,1)
		activations[layerID] = layer.AType
	
	# Generate node vectors and weight matrices
	for layerID in range(networkConfig.size()-1):
		var fromLayerSize = networkConfig[layerID].size
		var toLayerSize = networkConfig[layerID+1].size
		weights[ [layerID,layerID+1] ] = Matrix(toLayerSize,fromLayerSize,"random")
	getStatus("initial")

func test():
	var averageCost = 0
	for sampleID in testSetIDs:
		frontPropagate(dataset[sampleID].input)
		averageCost += getCost(dataset[sampleID].output)
	clearNodes()
	costHistory.append(averageCost/testSetIDs.size())
	getStatus("post-test")

func frontPropagate(sample:Array):
	clearNodes()
	# Load the sample into the input slot
	nodes[0] = sample
	var weightOrder = []
	for layerID in range(size-1):
		weightOrder.append( [layerID,layerID+1] )
	for layerIDPair in weightOrder:
		nodes[layerIDPair[1]] = matMult(weights[layerIDPair],nodes[layerIDPair[0]])
		nodes[layerIDPair[1]] = applyActivation(layerIDPair[1])
	#getStatus()

func applyActivation(layerID:int):
	var layer = nodes[layerID]
	var newLayer = []
	match activations[layerID]:
		AType.LINEAR:
			newLayer = layer
		AType.SOFTMAX:
			var eSum = 0
			for node in layer:
				eSum += exp(node[0])
			for node in layer:
				newLayer.append([exp(node[0])/eSum])
		AType.MAX:
			newLayer = layer
	return newLayer

func getCost(outputAnswer,outputLayer=nodes[size-1]):
	var cost = 0
	match costFunction:
		CostFunc.ABSOLUTE:
			for nodeID in range(outputLayer.size()):
				cost += abs(outputLayer[nodeID][0] - outputAnswer[nodeID][0])
		CostFunc.SQUARED:
			for nodeID in range(outputLayer.size()):
				cost += pow(outputLayer[nodeID][0] - outputAnswer[nodeID][0],2)
	return cost

func clearNodes():
	for layer in nodes:
		for node in layer:
			node = [0]

func Matrix(rows:int,cols:int,entries="zeros"):
	var matrix = []
	match entries:
		"zeros":
			for rowID in rows:
				var row = []
				for colID in cols:
					row.append(0)
				matrix.append(row)
		"random":
			for rowID in rows:
				var row = []
				for colID in cols:
					rng.randomize()
					row.append(rng.randf_range(0,1))
				matrix.append(row)
	return matrix

func getMatrixDim(matX:Array):
	var dim = {"rows":matX.size(),"cols":matX[0].size()}
	for row in matX:
		if row.size() != dim.cols:
			push_error("Invalid Matrix: "+str(matX))
	#print(dim)
	return dim

func matMult(matA:Array,matB:Array):
	var matA_dim = getMatrixDim(matA)
	var matB_dim = getMatrixDim(matB)
	if not matA_dim.cols == matB_dim.rows:
		push_error("Incompatible Matrices: "+str(matA)+" * "+str(matB))
	var matC = Matrix(matA_dim.rows,matB_dim.cols)
	for row in matC.size():
		for col in matC[0].size():
			for com in matA_dim.cols:
				matC[row][col] += matA[row][com]*matB[com][col]
	#print(matA," * ",matB," = ",matC)
	return matC





