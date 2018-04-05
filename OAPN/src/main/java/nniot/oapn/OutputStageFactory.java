package nniot.oapn;

import com.ociweb.pronghorn.neural.StageFactory;
import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;

public class OutputStageFactory implements StageFactory<MessageSchemaDynamic> {

    @Override
    public void newStage(GraphManager gm, Pipe<MessageSchemaDynamic> input, Pipe<MessageSchemaDynamic>[] output) {
        OutputStage stage = new OutputStage(gm, input, output);
    }

    @Override
    public void newStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic>[] output) {
        OutputStage stage = new OutputStage(gm, input, output);
    }

    @Override
    public void newStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic> output) {
        OutputStage stage = new OutputStage(gm, input, output);
    }
}