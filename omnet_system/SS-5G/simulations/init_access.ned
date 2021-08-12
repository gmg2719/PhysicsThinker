// MIT License
// 
// Copyright (c) 2021 PingzhouMing
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

package ss5g_dev.simulations;

@namespace(ss5G);

simple NrGnbBase
{
    parameters:
        double x_coord = default(10.0);
    	double y_coord = default(19.0);
    	double z_coord = default(6.0);
    	double period_sched = default(0.0005);		// The period of 500us
        @display("i=device/antennatower;bgb=1200,600");
    gates:
        input   in[];
        output out[];
        input   core_in;
        output  core_out;
}

simple NrUeBase
{
    parameters:
        double x_coord = default(0.0);
    	double y_coord = default(0.0);
    	double z_coord = default(0.0);
        @display("i=device/pocketpc;bgb=1200,600");
    gates:
        input   in;
        output out;
}

simple NrServer
{
    parameters:
    	@display("i=device/server;bgb=1200,600");
    gates:
    	input   in;
    	output out;
}

network indoor
{
    types:
        channel C extends ned.DelayChannel {
            delay = 10us;
            @display("ls=-");
        }
        channel D extends ned.DelayChannel {
            delay = 10ms;
        }
    submodules:
        gnb   : NrGnbBase;
        ue[2] : NrUeBase;
        server: NrServer;
    connections:
        ue[0].in <-- C <-- gnb.out++;
        ue[0].out --> C --> gnb.in++;
        ue[1].in <-- C <-- gnb.out++;
        ue[1].out --> C --> gnb.in++;
        server.in <-- D <-- gnb.core_out;
        server.out --> D --> gnb.core_in;
}

network outdoor
{
    types:
        channel C extends ned.DelayChannel {
            delay = 10us;
            @display("ls=-");
        }
        channel D extends ned.DelayChannel {
            delay = 10ms;
        }
    submodules:
        gnb   : NrGnbBase;
        ue[5] : NrUeBase;
        server: NrServer;
    connections:
        ue[0].in <-- C <-- gnb.out++;
        ue[0].out --> C --> gnb.in++;
        ue[1].in <-- C <-- gnb.out++;
        ue[1].out --> C --> gnb.in++;
        ue[2].in <-- C <-- gnb.out++;
        ue[2].out --> C --> gnb.in++;
        ue[3].in <-- C <-- gnb.out++;
        ue[3].out --> C --> gnb.in++;
        ue[4].in <-- C <-- gnb.out++;
        ue[4].out --> C --> gnb.in++;
        server.in <-- D <-- gnb.core_out;
        server.out --> D --> gnb.core_in;
}

