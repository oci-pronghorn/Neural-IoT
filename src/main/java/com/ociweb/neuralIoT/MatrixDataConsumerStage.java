package com.ociweb.neuralIoT;

import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import com.ociweb.pronghorn.stage.math.BuildMatrixCompute.MatrixTypes;
import com.ociweb.pronghorn.stage.math.MatrixSchema;
import com.ociweb.pronghorn.stage.math.RowSchema;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;

public class MatrixDataConsumerStage<M extends MatrixSchema<M>> extends PronghornStage {
	
	private final Pipe<RowSchema<M>> results;
	private MatrixSchema<M> schema;
	private final MatrixTypes type;
	
	public static <M extends MatrixSchema<M>> MatrixDataConsumerStage<M> newInstance(GraphManager gm, MatrixTypes type, Pipe<RowSchema<M>> results) {
		return new MatrixDataConsumerStage(gm,type,results);
	}
	
	public MatrixDataConsumerStage(GraphManager gm, MatrixTypes type, Pipe<RowSchema<M>> results) {
		super(gm, results, NONE);
		this.results = results;
		this.schema = results.config().schema().rootSchema();
		this.type = type;
		
	}

	@Override
	public void run() {
		
		int expectedRows = schema.rows;
		
		while (Pipe.hasContentToRead(results)) {
			//this is one row of the reults
						
			int msgIdx = Pipe.takeMsgIdx(results);
			int columns = schema.columns;
			
			if (MatrixTypes.Integers==type) {
				while (--columns>=0) {
					int aValue = Pipe.takeInt(results);
					//TODO: do something with this
					
				}
				
			}
			Pipe.confirmLowLevelRead(results, Pipe.sizeOf(results,msgIdx));
			Pipe.releaseReadLock(results);
						
		}
		
		
	}

}
