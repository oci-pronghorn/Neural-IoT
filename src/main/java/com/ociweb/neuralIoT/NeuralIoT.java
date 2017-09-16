package com.ociweb.neuralIoT;

import com.ociweb.pronghorn.neural.NeuralGraphBuilder;
import com.ociweb.pronghorn.neural.StageFactory;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.PipeConfig;
import com.ociweb.pronghorn.pipe.RawDataSchema;
import com.ociweb.pronghorn.stage.file.FileBlobReadStage;
import com.ociweb.pronghorn.stage.route.ReplicatorStage;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;
import com.ociweb.pronghorn.stage.scheduling.StageScheduler;
import com.ociweb.pronghorn.stage.test.ConsoleJSONDumpStage;
import com.ociweb.pronghorn.stage.test.PipeCleanerStage;
import com.ociweb.pronghorn.util.MainArgs;

public class NeuralIoT {

	public static void main(String[] args) {
		
		String inputFilePath = MainArgs.getOptArg("fileName", "-f", args, "./datafile.dat");
		
		GraphManager gm = new GraphManager();
		
		populateGraph(gm, inputFilePath);
		
		gm.enableTelemetry(8089);
		
		StageScheduler.defaultScheduler(gm).startup();
		
	}


	private static void populateGraph(GraphManager gm, String inputFilePath) {
				
		Pipe<RawDataSchema> pipe1= RawDataSchema.instance.newPipe(10, 10_000); //10 chunks each 10K in  size
		Pipe<RawDataSchema> pipe1A= RawDataSchema.instance.newPipe(20, 20_000); //10 chunks each 10K in  size
		Pipe<RawDataSchema> pipe1B= RawDataSchema.instance.newPipe(20, 20_000); //10 chunks each 10K in  size
		
		
		new FileBlobReadStage(gm, pipe1, inputFilePath); //This stage reads a file
		
		//This stage replicates the data to two pipes, great for debugging while passing on the real data.
		new ReplicatorStage<>(gm, pipe1, pipe1A, pipe1B); 
		
		new ConsoleJSONDumpStage(gm, pipe1A); //see all the data at the console.
		
		new PipeCleanerStage(gm, pipe1B); //dumps all data which came in 
		
				
	}

	public static void buildMatrixNeuralNet(GraphManager gm) {
		
		
		
	}
	
	public static void buildVisualNeuralNet(GraphManager gm) {

		int nodes2 = 0;
		
		PipeConfig config = null;
		StageFactory factory = new VisualStageFactory();
		
		
		Pipe[] prevA = null;
		NeuralGraphBuilder.buildPipeLayer(gm, config, prevA, nodes2, factory);

		Pipe[][] prevB = null;		
		NeuralGraphBuilder.buildPipeLayer(gm, config, prevB, nodes2, factory);
		
		Pipe[][] prevC = null;
		NeuralGraphBuilder.lastPipeLayer(gm, prevC, factory);
		
		
	}
	
	
}
