extends Control

onready var NeuralNetwork = $Network/NeuralNetwork

func _on_btnTest_pressed():
	NeuralNetwork.test()


func _on_btnShuffle_pressed():
	NeuralNetwork.generateTrainAndTest()


func _on_btnCreate_pressed():
	NeuralNetwork.createNetwork()

