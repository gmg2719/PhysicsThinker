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
#include "airframe_m.h"
#include "bscontrol_m.h"

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

typedef struct ue_info {
    int is_active;
    int sim_id;
    int in_port_index;
    double propagate_time;
} ue_info_t;

class NrUeBase : public cSimpleModule
{
  protected:
    int sim_id;
    int *state;
    double *bs_x_coord;
    double *bs_y_coord;
    double *bs_z_coord;
    double x_coord;
    double y_coord;
    double z_coord;
    long numSent;
    long numReceived;

  protected:
    virtual ~NrUeBase();

    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void forward2Bs_msg1(int port);
    virtual void forward2Bs_msg3(int port);
    virtual void forward2Bs_msg5(int port);

    // The finish() function is called by OMNeT++ at the end of the simulation:
    virtual void finish() override;
};

class NrGnbBase : public cSimpleModule
{
  protected:
    int sim_id;
    int state;
    long numSent;
    long numReceived;
    double x_coord;
    double y_coord;
    double z_coord;
    double period_sched;
    cMessage *control_msg;
    int numUeBacklog;
    std::vector<ue_info_t*> ue_backlog_table;

  protected:
    virtual ~NrGnbBase();
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;

    // Self scheduling
    virtual void broadcastPeriodMessage(BsControlMsg *msg);

    // Initial access
    virtual void forward_msg2(AirFrameMsg *ttmsg_ue);
    virtual void forward_msg4(AirFrameMsg *ttmsg_ue);

    ue_info_t* find_ue(int sim_id);
    int get_active_ue();
    int get_state() { return state; }

    // The finish() function is called by OMNeT++ at the end of the simulation:
    virtual void finish() override;
};

class NrPosition : public NrGnbBase
{
  private:
    long numSent;
    long numReceived;
    double period_positioning;
    cMessage *position_request_msg;

  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);

    // Send positioning request
    virtual void forward2core_positioning_request(AirFrameMsg *ttmsg_ue);
};

class NrSrsUe : public NrUeBase
{
  private:
    int bs_connected;
    cMessage *srs_control_event;

  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);

    virtual void srsPeriodForward();

    bool check_state();
};

};

#endif /* NRENTITY_H_ */
