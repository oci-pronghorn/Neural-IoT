package com.ociweb.neuralIoT;

import com.ociweb.pronghorn.neural.StageFactory;
import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;

public class VisualStageFactory implements StageFactory<MessageSchemaDynamic> {

	@Override
	public void newStage(GraphManager gm, Pipe<MessageSchemaDynamic> input, Pipe<MessageSchemaDynamic>[] output) {
		VisualNode stage = new VisualNode(gm, input, output);
		
		GraphManager.addNota(gm, GraphManager.SCHEDULE_RATE, 20_000, stage);
		
	}

	@Override
	public void newStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic>[] output) {
		VisualNode stage = new VisualNode(gm, input, output);
		
		GraphManager.addNota(gm, GraphManager.SCHEDULE_RATE, 20_000, stage);
	}
	
	@Override
	public void newStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic> output) {
		VisualNode stage = new VisualNode(gm, input, output);
		
		GraphManager.addNota(gm, GraphManager.SCHEDULE_RATE, 20_000, stage);
	}

}
