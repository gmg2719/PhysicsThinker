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

const double MY_PI = 3.1415926535897932384626438327950288;

const double SPEED_OF_LIGHT = 299792458.0;
const int SRS_CONFIG_TABLE[64][9] = {
        {0, 4, 1, 4, 1, 4, 1, 4, 1},
        {1, 8, 1, 4, 2, 4, 1, 4, 1},
        {2, 12, 1, 4, 3, 4, 1, 4, 1},
        {3, 16, 1, 4, 4, 4, 1, 4, 1},
        {4, 16, 1, 8, 2, 4, 2, 4, 1},
        {5, 20, 1, 4, 5, 4, 1, 4, 1},
        {6, 24, 1, 4, 6, 4, 1, 4, 1},
        {7, 24, 1, 12, 2, 4, 3, 4, 1},
        {8, 28, 1, 4, 7, 4, 1, 4, 1},
        {9, 32, 1, 16, 2, 8, 2, 4, 2},
        {10, 36, 1, 12, 3, 4, 3, 4, 1},
        {11, 40, 1, 20, 2, 4, 5, 4, 1},
        {12, 48, 1, 16, 3, 8, 2, 4, 2},
        {13, 48, 1, 24, 2, 12, 2, 4, 3},
        {14, 52, 1, 4, 13, 4, 1, 4, 1},
        {15, 56, 1, 28, 2, 4, 7, 4, 1},
        {16, 60, 1, 20, 3, 4, 5, 4, 1},
        {17, 64, 1, 32, 2, 16, 2, 4, 4},
        {18, 72, 1, 24, 3, 12, 2, 4, 3},
        {19, 72, 1, 36, 2, 12, 3, 4, 3},
        {20, 76, 1, 4, 19, 4, 1, 4, 1},
        {21, 80, 1, 40, 2, 20, 2, 4, 5},
        {22, 88, 1, 44, 2, 4, 11, 4, 1},
        {23, 96, 1, 32, 3, 16, 2, 4, 4},
        {24, 96, 1, 48, 2, 24, 2, 4, 6},
        {25, 104, 1, 52, 2, 4, 13, 4, 1},
        {26, 112, 1, 56, 2, 28, 2, 4, 7},
        {27, 120, 1, 60, 2, 20, 3, 4, 5},
        {28, 120, 1, 40, 3, 8, 5, 4, 2},
        {29, 120, 1, 24, 5, 12, 2, 4, 3},
        {30, 128, 1, 64, 2, 32, 2, 4, 8},
        {31, 128, 1, 64, 2, 32, 2, 4, 8},
        {32, 128, 1, 16, 8, 8, 2, 4, 2},
        {33, 132, 1, 44, 3, 4, 11, 4, 1},
        {34, 136, 1, 68, 2, 4, 17, 4, 1},
        {35, 144, 1, 72, 2, 36, 2, 4, 9},
        {36, 144, 1, 48, 3, 24, 2, 12, 2},
        {37, 144, 1, 48, 3, 16, 3, 4, 4},
        {38, 144, 1, 16, 9, 8, 2, 4, 2},
        {39, 152, 1, 76, 2, 4, 19, 4, 1},
        {40, 160, 1, 80, 2, 40, 2, 4, 10},
        {41, 160, 1, 80, 2, 20, 4, 4, 5},
        {42, 160, 1, 32, 5, 16, 2, 4, 4},
        {43, 168, 1, 84, 2, 28, 3, 4, 7},
        {44, 176, 1, 88, 2, 44, 2, 4, 11},
        {45, 184, 1, 92, 2, 4, 23, 4, 1},
        {46, 192, 1, 96, 2, 48, 2, 4, 12},
        {47, 192, 1, 96, 2, 24, 4, 4, 6},
        {48, 192, 1, 64, 3, 16, 4, 4, 4},
        {49, 192, 1, 24, 8, 8, 3, 4, 2},
        {50, 208, 1, 104, 2, 52, 2, 4, 13},
        {51, 216, 1, 108, 2, 36, 3, 4, 9},
        {52, 224, 1, 112, 2, 56, 2, 4, 14},
        {53, 240, 1, 120, 2, 60, 2, 4, 15},
        {54, 240, 1, 80, 3, 20, 4, 4, 5},
        {55, 240, 1, 48, 5, 16, 3, 8, 2},
        {56, 240, 1, 24, 10, 12, 2, 4, 3},
        {57, 256, 1, 128, 2, 64, 2, 4, 16},
        {58, 256, 1, 128, 2, 32, 4, 4, 8},
        {59, 256, 1, 16, 16, 8, 2, 4, 2},
        {60, 264, 1, 132, 2, 44, 3, 4, 11},
        {61, 272, 1, 136, 2, 68, 2, 4, 17},
        {62, 272, 1, 68, 4, 4, 17, 4, 1},
        {63, 272, 1, 16, 17, 8, 2, 4, 2}
};

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

enum NrWirelessChannelModel {
    INDOOR_ONLY_PATHLOSS,
    OUTDOOR_ONLY_PATHLOSS
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

typedef struct SrsPdu {
    int rnti;
    int bwp_size;
    int num_ports;
    int num_symbols;
    int bandwidth_index;
    int sequence_id;
    int group_or_sequence_hopping;
    int time_start_position;
    int cyclic_shift;
    int config_index;
    int comb_size;
    int comb_offset;
    int freq_shift;
    int freq_position;
    int freq_hopping;
    int resource_type;
    int t_srs;
}SrsPdu_t;

};

#endif /* NRENTITY_H_ */
