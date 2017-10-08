package com.ociweb.neuralIoT;

import com.ociweb.pronghorn.neural.NeuralGraphBuilder;
import com.ociweb.pronghorn.neural.StageFactory;
import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessFixedFieldPipeConfig;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;
import com.ociweb.pronghorn.stage.scheduling.StageScheduler;
import com.ociweb.pronghorn.util.MainArgs;

public class NeuralIoT {

	private static Appendable target;
	
	public static void main(String[] args) {
		
		//example to pull in a param
		String inputFilePath = MainArgs.getOptArg("fileName", "-f", args, "./datafile.dat");
		
		target = null;//System.out;
		
		GraphManager gm = new GraphManager();
		
		boolean useMatrix = false;
		if (useMatrix) {
			buildMatrixNeuralNet(gm);
		} else {
			buildVisualNeuralNet(gm);
		}
		
		//TODO: add switch to turn off line notations to build cleaner picture?
		
		gm.enableTelemetry(8089);
		
		StageScheduler.defaultScheduler(gm).startup();
		
	}


	public static void buildMatrixNeuralNet(GraphManager gm) {
		
		//simple matrix of ints just multipied in pipeline
		
		
	}
	
	public static void buildVisualNeuralNet(GraphManager gm) {

		
		final SchemalessFixedFieldPipeConfig config = new SchemalessFixedFieldPipeConfig(32);
		//config.hideLabels();
		
		final StageFactory<MessageSchemaDynamic> factory = new VisualStageFactory();
		
		
		int inputsCount = 5;		
		Pipe<MessageSchemaDynamic>[] prevA = Pipe.buildPipes(inputsCount, config);
		
		DataProducerStage prodStage = new DataProducerStage(gm, prevA);
		GraphManager.addNota(gm, GraphManager.SCHEDULE_RATE, 200_000, prodStage);	
		
		int nodesInLayerA = 3;
		Pipe<MessageSchemaDynamic>[][] fromA = NeuralGraphBuilder.buildPipeLayer(gm, config, prevA, nodesInLayerA, factory);

		int nodesInLayerB = 5;
		Pipe<MessageSchemaDynamic>[][] fromB = NeuralGraphBuilder.buildPipeLayer(gm, config, fromA, nodesInLayerB, factory);
		
		Pipe<MessageSchemaDynamic>[] fromC = NeuralGraphBuilder.lastPipeLayer(gm, fromB, factory);
			
		DataConsumerStage consStage = new DataConsumerStage(gm, fromC, target);
		GraphManager.addNota(gm, GraphManager.SCHEDULE_RATE, 10_000, consStage);
		
	}
	
	
}
