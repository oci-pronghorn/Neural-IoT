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


	public static <M extends MatrixSchema<M>> void buildMatrixNeuralNet(GraphManager gm) {
		
		
		MatrixTypes type = MatrixTypes.Integers;
		int leftRows=5;
		int rightColumns=3;
				
		int leftColumns = 2;
		int rightRows=leftColumns;		
				
		//walk leftRows , by rightCol for output
		//5x2
		//2x3
		
		int[][] leftTest = new int[][] {
			{1,2},
			{4,4},
			{7,7},
			{3,2},
			{1,1},
		}; 
		
		int[][] rightTest = new int[][] {
			{1,2,3},
			{4,4,4}
		}; 
	
		int targetThreadCount = 10;
		

		Pipe<RowSchema<M>> left = buildInputPipe(type, leftColumns, leftRows);		
		Pipe<RowSchema<M>> right = buildInputPipe(type, rightColumns, rightRows);
		
		//pipe populator here??
//		int rowId = resultSchema.rowId;
//		for(int c=0;c<leftRows;c++) {
//			while (!Pipe.hasRoomForWrite(left)) {
//				Thread.yield();
//			}
//			
//			Pipe.addMsgIdx(left, rowId);	
//			for(int r=0;r<leftTest[c].length;r++) {
//				type.addValue(leftTest[c][r], left);
//			}
//			Pipe.confirmLowLevelWrite(left, Pipe.sizeOf(left, rowId));
//			Pipe.publishWrites(left);
//			
//		}
		
		//buildSource(left, right);
		
		
		Pipe<RowSchema<M>> rowResults = BuildMatrixCompute.buildProductGraphRR(gm, targetThreadCount, left, right);
		
		/////////////////////////////////
		/////////////////////////////////
		
		writeToConsole(gm, rowResults);
		
	}


	private static <M extends MatrixSchema<M>> Pipe<RowSchema<M>> buildInputPipe(
			MatrixTypes type, 
			int columns,
			int rows) {
		
		return new Pipe<RowSchema<M>>(
										new PipeConfig<RowSchema<M>>(
											new RowSchema<M>(BuildMatrixCompute.buildSchema(rows, columns, type)), rows));
	
	}


	private static <M extends MatrixSchema<M>> void writeToConsole(GraphManager gm, Pipe<RowSchema<M>> rowResults) {
		MatrixSchema resultSchema = rowResults.config().schema().rootSchema();
		
		DecimalSchema<M> result2Schema = new DecimalSchema<M>(resultSchema);
		Pipe<DecimalSchema<M>> result2 = new Pipe<DecimalSchema<M>>(new PipeConfig<DecimalSchema<M>>(result2Schema, resultSchema.getRows()));
		ConvertToDecimalStage<M> watch = new ConvertToDecimalStage(gm, rowResults, result2);
		
		ConsoleJSONDumpStage.newInstance(gm,  result2, System.out);
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
