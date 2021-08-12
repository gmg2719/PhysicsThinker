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
#include "airframe_m.h"
#include "bscontrol_m.h"

namespace ss5G {

using namespace omnetpp;

enum NrMessageType {
    BS_CONTROL,
    BS_BROAD,
    BS_RRC_SETUP,
    UE_RRC_REQUEST,
    UE_RRC_COMPLETE
};

class NrUeBase : public cSimpleModule
{
  private:
    long numSent;
    long numReceived;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
};

class NrGnbBase : public cSimpleModule
{
  private:
    long numSent;
    long numReceived;
    double x_coord;
    double y_coord;
    double z_coord;
    double period_sched;
    cMessage *control_msg;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;

    // Self scheduling
    virtual void broadcastPeriodMessage(BsControlMsg *msg);

    // The finish() function is called by OMNeT++ at the end of the simulation:
    virtual void finish() override;
};

class NrUe : public NrUeBase
{
  private:
    double bs_x_coord;
    double bs_y_coord;
    double bs_z_coord;
    double x_coord;
    double y_coord;
    double z_coord;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
};

class NrGnb : public NrGnbBase
{
  private:

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
};

};

#endif /* NRENTITY_H_ */
