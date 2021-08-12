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

simple NrPosition extends NrGnbBase
{
    parameters:
        double period_positioning = default(0.02);    // The period of 20ms

	@class(NrPosition);  // Important!
}

network indoor_positioning
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
        gnb   : NrPosition;
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


