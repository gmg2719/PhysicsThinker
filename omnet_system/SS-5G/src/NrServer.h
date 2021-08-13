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
#ifndef NRSERVER_H_
#define NRSERVER_H_

#include <stdio.h>
#include <string.h>
#include <omnetpp.h>
#include <vector>
#include <map>
#include "corepacket_m.h"
#include "NrEntity.h"

namespace ss5G {

typedef struct ue_coordinate {
    double x;
    double y;
    double z;
} ue_coordinate_t;

typedef struct ue_position_info {
    int sim_id;
    // Utilize the 4 BSs data
    ue_coordinate_t bs_from[4];
    double arrive_time[4];
    std::vector<ue_coordinate_t> coord_est;
    std::vector<ue_coordinate_t> coord;
} ue_position_info_t;

class NrServer : public cSimpleModule
{
  protected:
    cMessage *position_do_event;
    std::map<int, ue_position_info_t> ue_position_table;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;

    void receive_positioning_request(CorePacket *coremsg);
    void tdoa_positioning();

    // The finish() function is called by OMNeT++ at the end of the simulation:
    virtual void finish() override;
};

};

#endif /* NRSERVER_H_ */
