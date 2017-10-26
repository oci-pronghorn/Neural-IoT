package com.ociweb.neuralIoT;

import com.ociweb.pronghorn.neural.NeuralGraphBuilder;
import com.ociweb.pronghorn.neural.StageFactory;
import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.PipeConfig;
import com.ociweb.pronghorn.pipe.SchemalessFixedFieldPipeConfig;
import com.ociweb.pronghorn.stage.math.BuildMatrixCompute;
import com.ociweb.pronghorn.stage.math.ConvertToDecimalStage;
import com.ociweb.pronghorn.stage.math.DecimalSchema;
import com.ociweb.pronghorn.stage.math.MatrixSchema;
import com.ociweb.pronghorn.stage.math.RowSchema;
import com.ociweb.pronghorn.stage.math.BuildMatrixCompute.MatrixTypes;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;
import com.ociweb.pronghorn.stage.scheduling.StageScheduler;
import com.ociweb.pronghorn.stage.test.ConsoleJSONDumpStage;
import com.ociweb.pronghorn.util.MainArgs;

public class NeuralIoT {

	private static Appendable target;
	
	public static void main(String[] args) {
		
		//example to pull in a param
		String inputFilePath = MainArgs.getOptArg("fileName", "-f", args, "./datafile.dat");
		
		target = null;//System.out;
		
		GraphManager gm = new GraphManager();
		
		GraphManager.addDefaultNota(gm, GraphManager.SCHEDULE_RATE, 20_000);
		
		boolean useMatrix = true;
		if (useMatrix) {
			buildMatrixNeuralNet(gm);
		} else {
			buildVisualNeuralNet(gm);
		}
		
		//TODO: add switch to turn off line notations to build cleaner picture?
		
		gm.enableTelemetry(8089);
		
		StageScheduler.defaultScheduler(gm).startup();
		
	}


	public static <M extends MatrixSchema<M>> void buildMatrixNeuralNet(GraphManager gm) {
		
		
		MatrixTypes type = MatrixTypes.Integers;
		int ttc = 1; //targetThreadCount, bump this up if you have more free cores..
	
		
		// primaryTrack     weights
		///////////////////////
		// 7x3          <-    3x5
		// 7x5          <-    5x3
		// 7x3          <-    3x5
		// 7x5    
	
		Pipe<RowSchema<M>> primary1 = buildInputPipe(type, 3,  7);		
		Pipe<RowSchema<M>> weights1 = buildInputPipe(type, 5, 3);
		
		Pipe<RowSchema<M>> primary2 = BuildMatrixCompute.buildProductGraphRR(gm, ttc, primary1, weights1);
		Pipe<RowSchema<M>> weights2 = buildInputPipe(type, 3, 5);
		
		Pipe<RowSchema<M>> primary3 = BuildMatrixCompute.buildProductGraphRR(gm, ttc, primary2, weights2);
		Pipe<RowSchema<M>> weights3 = buildInputPipe(type, 5, 3);				
		
		Pipe<RowSchema<M>> results = BuildMatrixCompute.buildProductGraphRR(gm, ttc, primary3, weights3);
		
		
		/////////////////////////////////
		/////////////////////////////////
		
		MatrixDataProducerStage.newInstance(gm, type, primary1);
		MatrixWeightsProducerStage.newInstance(gm, type, weights1);
		MatrixWeightsProducerStage.newInstance(gm, type, weights2);
		MatrixWeightsProducerStage.newInstance(gm, type, weights3);
		
		//TODO: replicator is in the pipeline but is that required??
		
		//TODO: no need to dump this a dedicated stage to consume would be better
		writeToConsole(gm, results);
		
        //TODO: the column converters need a flag to hold and re-use the last value so we only need to xmit once....
		

		
	}


	private static <M extends MatrixSchema<M>> Pipe<RowSchema<M>> buildInputPipe(
			MatrixTypes type, 
			int columns,
			int rows) {
		
		PipeConfig<RowSchema<M>> config = new PipeConfig<RowSchema<M>>(
					new RowSchema<M>(BuildMatrixCompute.buildSchema(rows, columns, type)), rows);
		
		config.hideLabels();
		
		return new Pipe<RowSchema<M>>(
										  config);
	
	}


	private static <M extends MatrixSchema<M>> void writeToConsole(GraphManager gm, Pipe<RowSchema<M>> rowResults) {
		MatrixSchema resultSchema = rowResults.config().schema().rootSchema();
		
		DecimalSchema<M> result2Schema = new DecimalSchema<M>(resultSchema);
		PipeConfig<DecimalSchema<M>> config = new PipeConfig<DecimalSchema<M>>(result2Schema, resultSchema.getRows());
		config.hideLabels();
		Pipe<DecimalSchema<M>> result2 = new Pipe<DecimalSchema<M>>(config);
		ConvertToDecimalStage<M> watch = new ConvertToDecimalStage(gm, rowResults, result2);
			
		ConsoleJSONDumpStage dump = ConsoleJSONDumpStage.newInstance(gm,  result2, System.out);
	}


	public static void buildVisualNeuralNet(GraphManager gm) {

		
		final SchemalessFixedFieldPipeConfig config = new SchemalessFixedFieldPipeConfig(32);
		config.hideLabels();
		
		final StageFactory<MessageSchemaDynamic> factory = new VisualStageFactory();
		
		
		int inputsCount = 5;		
		Pipe<MessageSchemaDynamic>[] prevA = Pipe.buildPipes(inputsCount, config);
		
		DataProducerStage.newInstance(gm, prevA);
			
		int nodesInLayerA = 3;
		Pipe<MessageSchemaDynamic>[][] fromA = NeuralGraphBuilder.buildPipeLayer(gm, config, prevA, nodesInLayerA, factory);

		int nodesInLayerB = 5;
		Pipe<MessageSchemaDynamic>[][] fromB = NeuralGraphBuilder.buildPipeLayer(gm, config, fromA, nodesInLayerB, factory);
		
		Pipe<MessageSchemaDynamic>[] fromC = NeuralGraphBuilder.lastPipeLayer(gm, fromB, factory);
			
		DataConsumerStage.newInstance(gm, fromC, target);

		
	}
	
	
}
