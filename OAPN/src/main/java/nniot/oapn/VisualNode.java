package nniot.oapn;

import java.util.Arrays;

import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessPipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;

public class VisualNode extends PronghornStage {

	private final Pipe<MessageSchemaDynamic>[] input; 
	private final Pipe<MessageSchemaDynamic>[] output;
	private float[] weights;
	
	public VisualNode(GraphManager gm, Pipe<MessageSchemaDynamic> input, Pipe<MessageSchemaDynamic>[] output) {
		super(gm, input, output);
		this.input = new Pipe[]{input};
		this.output = output;
		buildWeights();
	}

	public VisualNode(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic>[] output) {
		super(gm, input, output);
		this.input = input;
		this.output = output;
		buildWeights();
	}

	public VisualNode(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic> output) {
		super(gm, input, output);
		this.input = input;
		this.output = new Pipe[]{output};
		buildWeights();
	}

	private void buildWeights() {		
		//TODO: should ask a singleton dictionary for the weights...
		
		this.weights = new float[this.input.length];
		Arrays.fill(weights, 1);
		
	}
	
	@Override
	public void run() {
				
		int c = 0;
		while ((c>0) || ((c=availCount())>0) ) {
			c -= 1;//this 1 represents one 32 bit integer
			
			float result = 0;
			int i = input.length;
			while (--i>=0) {				
				
				float value = SchemalessPipe.readFloat(input[i]);
				SchemalessPipe.releaseReads(input[i]);
				
				
				//TODO: rework this?
				result += (value*weights[i]);			
			}
			
			//send this value to all the down stream nodes
			
			int j = output.length;
			while (--j>=0) {
				
				SchemalessPipe.writeFloat(output[j], result);
				SchemalessPipe.publishWrites(output[j]);
				
				
			}			
		}				
	}

	private int availCount() {
		int avail = messagesToConsume();
		if (avail>0) {
			avail = Math.min(avail, messagesOutputRoom());
		}
		return avail;
	}

	private int messagesToConsume() {
		
		int results = Integer.MAX_VALUE;
		int i = input.length;
		assert(i>0);
		while (--i>=0) {
			results = Math.min(results, SchemalessPipe.contentRemaining(input[i]));
		}
		return results;
	}

	private int messagesOutputRoom() {
		
		int results = Integer.MAX_VALUE;
		int i = output.length;
		assert(i>0);
		while (--i>=0) {
			results = Math.min(results, 
						       SchemalessPipe.roomRemaining(output[i])
					);
		}
		return results;
	}
	
}
