package com.ociweb.neuralIoT;

import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessPipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;

public class DataProducerStage extends PronghornStage {

	private final Pipe<MessageSchemaDynamic>[] output;
	
	public DataProducerStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] output) {
		super(gm, NONE, output);
		this.output = output;
	}

	@Override
	public void run() {
		
		int c = 0;
		
		while (c>0 || ((c = roomForWrite()) > 0) ){
			c -= 1;

			int i = output.length;
			while (--i>=0) {
				
				float someValue = 1;
				
				SchemalessPipe.writeFloat(output[i], someValue);
				SchemalessPipe.publishWrites(output[i]);
				
			}
		}		
	}
	
	public int roomForWrite() {
		int result = Integer.MAX_VALUE;
		int i = output.length;
		while (--i>=0) {
			result = Math.min(result, SchemalessPipe.roomRemaining(output[i]));
		}
		return result;
	}
	

}
