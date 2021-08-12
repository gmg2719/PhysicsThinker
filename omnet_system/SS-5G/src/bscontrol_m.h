//
// Generated file, do not edit! Created by nedtool 5.6 from src/bscontrol.msg.
//

#ifndef __BSCONTROL_M_H
#define __BSCONTROL_M_H

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wreserved-id-macro"
#endif
#include <omnetpp.h>

// nedtool version check
#define MSGC_VERSION 0x0506
#if (MSGC_VERSION!=OMNETPP_VERSION)
#    error Version mismatch! Probably this file was generated by an earlier version of nedtool: 'make clean' should help.
#endif



/**
 * Class generated from <tt>src/bscontrol.msg:3</tt> by nedtool.
 * <pre>
 * message BsControlMsg
 * {
 *     int timer;
 * }
 * </pre>
 */
class BsControlMsg : public ::omnetpp::cMessage
{
  protected:
    int timer;

  private:
    void copy(const BsControlMsg& other);

  protected:
    // protected and unimplemented operator==(), to prevent accidental usage
    bool operator==(const BsControlMsg&);

  public:
    BsControlMsg(const char *name=nullptr, short kind=0);
    BsControlMsg(const BsControlMsg& other);
    virtual ~BsControlMsg();
    BsControlMsg& operator=(const BsControlMsg& other);
    virtual BsControlMsg *dup() const override {return new BsControlMsg(*this);}
    virtual void parsimPack(omnetpp::cCommBuffer *b) const override;
    virtual void parsimUnpack(omnetpp::cCommBuffer *b) override;

    // field getter/setter methods
    virtual int getTimer() const;
    virtual void setTimer(int timer);
};

inline void doParsimPacking(omnetpp::cCommBuffer *b, const BsControlMsg& obj) {obj.parsimPack(b);}
inline void doParsimUnpacking(omnetpp::cCommBuffer *b, BsControlMsg& obj) {obj.parsimUnpack(b);}


#endif // ifndef __BSCONTROL_M_H

