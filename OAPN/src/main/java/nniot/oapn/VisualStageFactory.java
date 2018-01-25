package nniot.oapn;

import com.ociweb.pronghorn.neural.StageFactory;
import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;
import java.util.HashMap;

public class VisualStageFactory implements StageFactory<MessageSchemaDynamic> {
        
	@Override
	public void newStage(GraphManager gm, Pipe<MessageSchemaDynamic> input, Pipe<MessageSchemaDynamic>[] output) {
		VisualNode stage = new VisualNode(gm, input, output);
				
	}

	@Override
	public void newStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic>[] output) {
		VisualNode stage = new VisualNode(gm, input, output);
		
	}
	
	@Override
	public void newStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic> output) {
		VisualNode stage = new VisualNode(gm, input, output);
		
	}

}
