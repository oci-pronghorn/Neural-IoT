package com.ociweb.neuralIoT;

import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import com.ociweb.pronghorn.stage.math.MatrixSchema;
import com.ociweb.pronghorn.stage.math.RowSchema;
import com.ociweb.pronghorn.stage.math.BuildMatrixCompute.MatrixTypes;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;

public class MatrixWeightsProducerStage<M extends MatrixSchema<M>> extends PronghornStage {

	private final Pipe<RowSchema<M>> weights;
	private final int rowMsgId;
	private final int rows;
	private final int cols;
	private final MatrixTypes type;
	private int activeColumn;
	
	public static <M extends MatrixSchema<M>> MatrixWeightsProducerStage<M> newInstance(GraphManager gm, MatrixTypes type, Pipe<RowSchema<M>> weights) {
		return new MatrixWeightsProducerStage<M>(gm, type, weights);
	}
	
	public MatrixWeightsProducerStage(GraphManager gm, MatrixTypes type, Pipe<RowSchema<M>> weights) {
		super(gm, NONE, weights);
		this.weights = weights;
		
		MatrixSchema<M> rootSchema = ((RowSchema<M>) weights.config().schema()).rootSchema();
		
		this.rowMsgId = rootSchema.rowId;
		this.rows = rootSchema.rows;
		this.cols = rootSchema.columns;
		this.type = type;
	}

	@Override
	public void startup() {
		activeColumn = 0; //send value
	}
	
	@Override
	public void run() {

		while (activeColumn<cols && Pipe.hasRoomForWrite(weights)) {
								
			Pipe.addMsgIdx(weights, rowMsgId);	
			for(int r=0;r<rows;r++) {
				
				//some weights for this row..
				//TODO: using activeColumn and r send the weight
				int value = 1;
				
				type.addValue(value, weights);
			
			}
			Pipe.confirmLowLevelWrite(weights, Pipe.sizeOf(weights, rowMsgId));
			Pipe.publishWrites(weights);
			
			activeColumn++;
			
		}
		if (activeColumn == cols) {
			activeColumn = 0;//send value again
		}
		
		
	}
}
