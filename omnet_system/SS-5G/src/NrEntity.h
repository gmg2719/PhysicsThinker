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

#ifndef NRENTITY_H_
#define NRENTITY_H_

#include <stdio.h>
#include <string.h>
#include <omnetpp.h>
#include <vector>

namespace ss5G {

using namespace omnetpp;

const double SPEED_OF_LIGHT = 299792458.0;

enum NrMessageType {
    BS_BROAD,
    BS_MSG2,
    BS_MSG4,
    BS_POSITION_REQ,
    UE_MSG1,
    UE_MSG3,
    // COMPLETE_RRC is equal to the msg5 in the 5G NR standard
    UE_COMPLETE_RRC,
    UE_SRS_SIGNAL
};

enum NrControlSignalType {
    INIT_BROADCAST,
    POSITION_REQUEST,
    POSITION_DO
};

enum GnbState {
    SWITCH_ON_STATE,
    READY_STATE,
};

enum UeState {
    RRC_IDLE,
    RRC_SETUP,
    RRC_CONNECTED
};

};

#endif /* NRENTITY_H_ */