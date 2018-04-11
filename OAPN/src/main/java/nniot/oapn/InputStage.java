package nniot.oapn;

import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessPipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import static com.ociweb.pronghorn.stage.PronghornStage.NONE;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;

public class InputStage extends PronghornStage {

    private float data;
    private final Pipe<MessageSchemaDynamic>[] output;

    public static InputStage newInstance(GraphManager gm, Pipe<MessageSchemaDynamic>[] output) {
        return new InputStage(gm, output);
    }

    public InputStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] output) {
        super(gm, NONE, output);
        this.output = output;
        this.data = Float.NaN;
    }

    //Hands floats out to pipes below it
    public void run() {
        if (!(this.data == Float.NaN))  {
            int c = 0;
            while (c > 0 || ((c = roomForWrite()) > 0)) {
                c -= 1;

                int i = output.length;
                while (--i >= 0) {
                    SchemalessPipe.writeFloat(output[i], data);
                    SchemalessPipe.publishWrites(output[i]);
                }
            }
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

    public void giveInputData(float data) {
        this.data = data;
    }
}
