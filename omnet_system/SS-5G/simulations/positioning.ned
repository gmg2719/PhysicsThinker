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
// Import other useful NED modules
import ss5g_dev.channels.NrChannel;

@namespace(ss5G);

simple NrPosition extends NrGnbBase
{
    parameters:
        double period_positioning = default(0.02);    // The period of 20ms

	@class(NrPosition);  // Important!
}

simple NrSrsUe extends NrUeBase
{
    @class(NrSrsUe);
}

network indoor_positioning
{
    types:
        channel D extends ned.DelayChannel {
            delay = 10ms;
        }
    submodules:
        gnb1  : NrPosition;
        gnb2  : NrPosition;
        gnb3  : NrPosition;
        gnb4  : NrPosition;
        wireless_channel : NrChannel; 
        ue    : NrSrsUe;
        server: NrServer;
    connections:
        ue.in++ <-- wireless_channel.out++;
        wireless_channel.in++ <-- gnb1.out++;
        ue.out++ --> wireless_channel.in++;
        wireless_channel.out++ --> gnb1.in++;
        
        ue.in++ <-- wireless_channel.out++;
        wireless_channel.in++ <-- gnb2.out++;
        ue.out++ --> wireless_channel.in++;
        wireless_channel.out++ --> gnb2.in++;
        
        ue.in++ <-- wireless_channel.out++;
        wireless_channel.in++ <-- gnb3.out++;
        ue.out++ --> wireless_channel.in++;
        wireless_channel.out++ --> gnb3.in++;
        
        ue.in++ <-- wireless_channel.out++;
        wireless_channel.in++ <-- gnb4.out++;
        ue.out++ --> wireless_channel.in++;
        wireless_channel.out++ --> gnb4.in++;
        server.in++ <-- D <-- gnb1.core_out;
        server.out++ --> D --> gnb1.core_in;
        server.in++ <-- D <-- gnb2.core_out;
        server.out++ --> D --> gnb2.core_in;
        server.in++ <-- D <-- gnb3.core_out;
        server.out++ --> D --> gnb3.core_in;
        server.in++ <-- D <-- gnb4.core_out;
        server.out++ --> D --> gnb4.core_in;
}


