package com.ociweb.neuralIoT;

import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessPipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;
import com.ociweb.pronghorn.util.AppendableProxy;
import com.ociweb.pronghorn.util.Appendables;

public class DataConsumerStage extends PronghornStage {

	private final Pipe<MessageSchemaDynamic>[] input;
	private AppendableProxy target;
	
	public static DataConsumerStage newInstance(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Appendable target) {
		return new DataConsumerStage(gm, input, target);
	}
	
	public DataConsumerStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Appendable target) {
		super(gm,input,NONE);
		this.input = input;
		this.target = null==target?null:Appendables.wrap(target);
	}

	@Override
	public void run() {
	
		int c = 0;		
		while (c>0 || ((c = contentToRead()) > 0)) {
			c -= 1;
			
			int i = input.length;
			while (--i>=0) {
				
				float value = SchemalessPipe.readFloat(input[i]);
				SchemalessPipe.releaseReads(input[i]);
		
				if (null!=target) {
					target.append(Float.toString(value)).append(", ");
				}
			}
			if (null!=target) {
				target.append("\n");
			}
			
		}
	}

	public int contentToRead() {
		int result = Integer.MAX_VALUE;
		int i = input.length;
		while (--i>=0) {
			result = Math.min(result, SchemalessPipe.contentRemaining(input[i]));
		}
		return result;
	}
	
}
