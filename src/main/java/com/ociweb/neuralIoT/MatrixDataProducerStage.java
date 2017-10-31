package com.ociweb.neuralIoT;

import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import com.ociweb.pronghorn.stage.math.MatrixSchema;
import com.ociweb.pronghorn.stage.math.RowSchema;
import com.ociweb.pronghorn.stage.math.BuildMatrixCompute.MatrixTypes;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;

public class MatrixDataProducerStage<M extends MatrixSchema<M>> extends PronghornStage {

	private final Pipe<RowSchema<M>> primary;
	private final MatrixTypes type;
	private int activeColumn;
	private final int rowMsgId;
	private final int rows;
	private final int cols;
	
	public static <M extends MatrixSchema<M>> MatrixDataProducerStage<M> newInstance(GraphManager gm, MatrixTypes type, Pipe<RowSchema<M>> primary) {
		return new MatrixDataProducerStage<M>(gm, type, primary);
	}
	
	public MatrixDataProducerStage(GraphManager gm, MatrixTypes type, Pipe<RowSchema<M>> primary) {
		super(gm, NONE, primary);
		this.primary = primary;
		this.type = type;
		
		MatrixSchema<M> rootSchema = ((RowSchema<M>) primary.config().schema()).rootSchema();
		
		this.rowMsgId = rootSchema.rowId;
		this.rows = rootSchema.rows;
		this.cols = rootSchema.columns;
	}


	@Override
	public void startup() {
		activeColumn = 0; //send value
	}
	
	@Override
	public void run() {

		while (activeColumn<cols && Pipe.hasRoomForWrite(primary)) {
								
			Pipe.addMsgIdx(primary, rowMsgId);	
			for(int r=0;r<rows;r++) {
				
				//some primarys for this row..
				//TODO: using activeColumn and r send the primary input value
				int value = 1;
				
				type.addValue(value, primary);
			
			}
			Pipe.confirmLowLevelWrite(primary, Pipe.sizeOf(primary, rowMsgId));
			Pipe.publishWrites(primary);
			
			activeColumn++;
			
		}
		if (activeColumn == cols) {
			activeColumn = 0;//send value again
		}
		
		
	}

}
