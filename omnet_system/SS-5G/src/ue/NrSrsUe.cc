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

#include "NrSrsUe.h"

using namespace omnetpp;

namespace ss5G {

Define_Module(NrSrsUe);

NrSrsUe::~NrSrsUe()
{
    if (srs_control_event != NULL)
        delete srs_control_event;
}

void NrSrsUe::initialize()
{
    bs_connected = gateSize("out");

    NrUeBase::initialize();
    tx_ants = 1;
    srs_pdu_initialize();

    // Set the channel model from the configuration file
    channel_model = par("channel_model");

    char msgname[32];
    sprintf(msgname, "srs-peirod_ind");
    cMessage *msg = new cMessage(msgname);
    srs_control_event = msg;

    scheduleAt(5.0, msg);
    EV << "UE prepare to do SRS period indication, next 5 seconds !\n";
}

void NrSrsUe::handleMessage(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        if (check_state()) {
            srsPeriodForward();
            update_positions();

            EV << "UE do SRS period indication, next 100 ms !\n";
            scheduleAt(simTime() + 0.01, msg);
        } else {
            EV << "UE can not do SRS period indication, wait 5 seconds !\n";
            scheduleAt(simTime() + 5.0, msg);
        }
    } else {
        NrUeBase::handleMessage(msg);
    }
}

void NrSrsUe::srs_pdu_initialize()
{
    srs_config.rnti = 51378;
    srs_config.bwp_size = 273;
    srs_config.num_ports = 1;
    srs_config.num_symbols = 1;
    srs_config.bandwidth_index = 0;
    srs_config.sequence_id = 0;
    srs_config.group_or_sequence_hopping = 0;
    srs_config.time_start_position = 13;
    srs_config.cyclic_shift = 0;
    srs_config.config_index = 63;
    srs_config.comb_size = 2;
    srs_config.comb_offset = 0;
    srs_config.freq_shift = 0;
    srs_config.freq_position = 0;
    srs_config.freq_hopping = 0;
    srs_config.resource_type = 2;
    srs_config.t_srs = 1;

    srs_freq_position_calc();

    num_srs_re = num_rb * 12 / srs_config.comb_size;
    cs_max = srs_config.comb_size == 4 ? 8 : 12;

    num_re_total = srs_config.bwp_size * 12;
}

void NrSrsUe::srs_freq_position_calc()
{
    int c_srs = srs_config.config_index;
    int b_srs = srs_config.bandwidth_index;
    int comb_size = srs_config.comb_size;
    int comb_offset = srs_config.comb_offset;
    int n_shift = srs_config.freq_shift;
    int n_rrc = srs_config.freq_position;

    num_rb = SRS_CONFIG_TABLE[c_srs][2*b_srs + 1];

    int n = SRS_CONFIG_TABLE[c_srs][2*b_srs + 2];
    int k0_bar = n_shift * 12 + comb_offset;
    k0 = k0_bar;
    for (int b = 0; b < b_srs; b++) {
        k0 += n_shift + num_rb * comb_size * ((4 *n_rrc/num_rb) % n);
    }
}

void NrSrsUe::update_positions()
{
    if (bs_connected < 4) {
        return;
    }

    for (int i = 0; i < 4; i++) {
        ue2bs_dist[i] = sqrt((x_coord-bs_x_coord[i])*(x_coord-bs_x_coord[i])+(y_coord-bs_y_coord[i])*(y_coord-bs_y_coord[i])+
                (z_coord-bs_z_coord[i])*(z_coord-bs_z_coord[i]));
    }
}

void NrSrsUe::srsPeriodForward()
{
    int src = NrUeBase::sim_id;

    char msgname[64];
    for (int i = 0; i < bs_connected; i++) {
        // Create message object and set source and destination field.
        sprintf(msgname, "ue-%d-srs-ind", src);
        AirFrameMsg *msg = new AirFrameMsg(msgname);
        msg->setSource(src);
        msg->setDestination(-1);
        // Set the message type
        msg->setType(UE_SRS_SIGNAL);
        // Set the coordinates of the UE and the BS destination
        msg->setX(x_coord);
        msg->setY(y_coord);
        msg->setZ(z_coord);
        msg->setDestX(bs_x_coord[i]);
        msg->setDestY(bs_y_coord[i]);
        msg->setDestZ(bs_z_coord[i]);
        // Set the power of the SRS period signal, fixed 40 dB, center_frequency = 2.6 GHz
        msg->setTxPowerUpdate(40.0);
        msg->setCenterFreq(2.6);
        msg->setChannelMode(channel_model);
        // Set the time information
        msg->setTimeStamp(0.0);
        EV << "Broadcasting message " << msg << " on BS out[" << i << "]\n";
        send(msg, "out", i);
        numSent++;
    }
}

bool NrSrsUe::check_state()
{
    if (bs_connected <= 0) {
        return false;
    }

    for (int i = 0; i < bs_connected; i++) {
        if (NrUeBase::state[i] != RRC_CONNECTED) {
            return false;
        }
    }

    return true;
}

};

