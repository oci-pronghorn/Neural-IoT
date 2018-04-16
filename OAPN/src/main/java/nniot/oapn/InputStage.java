package nniot.oapn;

import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessPipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import static com.ociweb.pronghorn.stage.PronghornStage.NONE;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;

public class InputStage extends PronghornStage {

    private float[] data;
    private final Pipe<MessageSchemaDynamic>[] output;

    public static InputStage newInstance(GraphManager gm, Pipe<MessageSchemaDynamic>[] output) {
        return new InputStage(gm, output);
    }

    public InputStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] output) {
        super(gm, NONE, output);
        this.output = output;
        this.data = null;
    }

    //Hands floats out to pipes below it
    @Override
    public void run() {
        if (this.data != null && roomForWrite() > 0) {
            for (int i = 0; i < output.length; i++) {
                SchemalessPipe.writeFloat(output[i], data[i]);
                SchemalessPipe.publishWrites(output[i]);
            }

            this.data = null;
        }
    }

    public int roomForWrite() {
        int result = Integer.MAX_VALUE;
        int i = output.length;
        while (--i >= 0) {
            result = Math.min(result, SchemalessPipe.roomRemaining(output[i]));
        }
        return result;
    }

    public void giveInputData(float[] data) {
        this.data = data;
    }
}
