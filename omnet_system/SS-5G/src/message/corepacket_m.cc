//
// Generated file, do not edit! Created by nedtool 5.6 from message/corepacket.msg.
//

// Disable warnings about unused variables, empty switch stmts, etc:
#ifdef _MSC_VER
#  pragma warning(disable:4101)
#  pragma warning(disable:4065)
#endif

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wshadow"
#  pragma clang diagnostic ignored "-Wconversion"
#  pragma clang diagnostic ignored "-Wunused-parameter"
#  pragma clang diagnostic ignored "-Wc++98-compat"
#  pragma clang diagnostic ignored "-Wunreachable-code-break"
#  pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#  pragma GCC diagnostic ignored "-Wshadow"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wold-style-cast"
#  pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#  pragma GCC diagnostic ignored "-Wfloat-conversion"
#endif

#include <iostream>
#include <sstream>
#include "corepacket_m.h"

namespace omnetpp {

// Template pack/unpack rules. They are declared *after* a1l type-specific pack functions for multiple reasons.
// They are in the omnetpp namespace, to allow them to be found by argument-dependent lookup via the cCommBuffer argument

// Packing/unpacking an std::vector
template<typename T, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::vector<T,A>& v)
{
    int n = v.size();
    doParsimPacking(buffer, n);
    for (int i = 0; i < n; i++)
        doParsimPacking(buffer, v[i]);
}

template<typename T, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::vector<T,A>& v)
{
    int n;
    doParsimUnpacking(buffer, n);
    v.resize(n);
    for (int i = 0; i < n; i++)
        doParsimUnpacking(buffer, v[i]);
}

// Packing/unpacking an std::list
template<typename T, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::list<T,A>& l)
{
    doParsimPacking(buffer, (int)l.size());
    for (typename std::list<T,A>::const_iterator it = l.begin(); it != l.end(); ++it)
        doParsimPacking(buffer, (T&)*it);
}

template<typename T, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::list<T,A>& l)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i=0; i<n; i++) {
        l.push_back(T());
        doParsimUnpacking(buffer, l.back());
    }
}

// Packing/unpacking an std::set
template<typename T, typename Tr, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::set<T,Tr,A>& s)
{
    doParsimPacking(buffer, (int)s.size());
    for (typename std::set<T,Tr,A>::const_iterator it = s.begin(); it != s.end(); ++it)
        doParsimPacking(buffer, *it);
}

template<typename T, typename Tr, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::set<T,Tr,A>& s)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i=0; i<n; i++) {
        T x;
        doParsimUnpacking(buffer, x);
        s.insert(x);
    }
}

// Packing/unpacking an std::map
template<typename K, typename V, typename Tr, typename A>
void doParsimPacking(omnetpp::cCommBuffer *buffer, const std::map<K,V,Tr,A>& m)
{
    doParsimPacking(buffer, (int)m.size());
    for (typename std::map<K,V,Tr,A>::const_iterator it = m.begin(); it != m.end(); ++it) {
        doParsimPacking(buffer, it->first);
        doParsimPacking(buffer, it->second);
    }
}

template<typename K, typename V, typename Tr, typename A>
void doParsimUnpacking(omnetpp::cCommBuffer *buffer, std::map<K,V,Tr,A>& m)
{
    int n;
    doParsimUnpacking(buffer, n);
    for (int i=0; i<n; i++) {
        K k; V v;
        doParsimUnpacking(buffer, k);
        doParsimUnpacking(buffer, v);
        m[k] = v;
    }
}

// Default pack/unpack function for arrays
template<typename T>
void doParsimArrayPacking(omnetpp::cCommBuffer *b, const T *t, int n)
{
    for (int i = 0; i < n; i++)
        doParsimPacking(b, t[i]);
}

template<typename T>
void doParsimArrayUnpacking(omnetpp::cCommBuffer *b, T *t, int n)
{
    for (int i = 0; i < n; i++)
        doParsimUnpacking(b, t[i]);
}

// Default rule to prevent compiler from choosing base class' doParsimPacking() function
template<typename T>
void doParsimPacking(omnetpp::cCommBuffer *, const T& t)
{
    throw omnetpp::cRuntimeError("Parsim error: No doParsimPacking() function for type %s", omnetpp::opp_typename(typeid(t)));
}

template<typename T>
void doParsimUnpacking(omnetpp::cCommBuffer *, T& t)
{
    throw omnetpp::cRuntimeError("Parsim error: No doParsimUnpacking() function for type %s", omnetpp::opp_typename(typeid(t)));
}

}  // namespace omnetpp


// forward
template<typename T, typename A>
std::ostream& operator<<(std::ostream& out, const std::vector<T,A>& vec);

// Template rule which fires if a struct or class doesn't have operator<<
template<typename T>
inline std::ostream& operator<<(std::ostream& out,const T&) {return out;}

// operator<< for std::vector<T>
template<typename T, typename A>
inline std::ostream& operator<<(std::ostream& out, const std::vector<T,A>& vec)
{
    out.put('{');
    for(typename std::vector<T,A>::const_iterator it = vec.begin(); it != vec.end(); ++it)
    {
        if (it != vec.begin()) {
            out.put(','); out.put(' ');
        }
        out << *it;
    }
    out.put('}');
    
    char buf[32];
    sprintf(buf, " (size=%u)", (unsigned int)vec.size());
    out.write(buf, strlen(buf));
    return out;
}

Register_Class(CorePacket)

CorePacket::CorePacket(const char *name, short kind) : ::omnetpp::cPacket(name,kind)
{
    this->type = 0;
    this->bsSimId = 0;
    this->ueSimId = 0;
    this->bsX = 0;
    this->bsY = 0;
    this->bsZ = 0;
    this->ueX = 0;
    this->ueY = 0;
    this->ueZ = 0;
    this->arriveTime = 0;
    this->timeStamp = 0;
}

CorePacket::CorePacket(const CorePacket& other) : ::omnetpp::cPacket(other)
{
    copy(other);
}

CorePacket::~CorePacket()
{
}

CorePacket& CorePacket::operator=(const CorePacket& other)
{
    if (this==&other) return *this;
    ::omnetpp::cPacket::operator=(other);
    copy(other);
    return *this;
}

void CorePacket::copy(const CorePacket& other)
{
    this->type = other.type;
    this->bsSimId = other.bsSimId;
    this->ueSimId = other.ueSimId;
    this->bsX = other.bsX;
    this->bsY = other.bsY;
    this->bsZ = other.bsZ;
    this->ueX = other.ueX;
    this->ueY = other.ueY;
    this->ueZ = other.ueZ;
    this->arriveTime = other.arriveTime;
    this->timeStamp = other.timeStamp;
}

void CorePacket::parsimPack(omnetpp::cCommBuffer *b) const
{
    ::omnetpp::cPacket::parsimPack(b);
    doParsimPacking(b,this->type);
    doParsimPacking(b,this->bsSimId);
    doParsimPacking(b,this->ueSimId);
    doParsimPacking(b,this->bsX);
    doParsimPacking(b,this->bsY);
    doParsimPacking(b,this->bsZ);
    doParsimPacking(b,this->ueX);
    doParsimPacking(b,this->ueY);
    doParsimPacking(b,this->ueZ);
    doParsimPacking(b,this->arriveTime);
    doParsimPacking(b,this->timeStamp);
}

void CorePacket::parsimUnpack(omnetpp::cCommBuffer *b)
{
    ::omnetpp::cPacket::parsimUnpack(b);
    doParsimUnpacking(b,this->type);
    doParsimUnpacking(b,this->bsSimId);
    doParsimUnpacking(b,this->ueSimId);
    doParsimUnpacking(b,this->bsX);
    doParsimUnpacking(b,this->bsY);
    doParsimUnpacking(b,this->bsZ);
    doParsimUnpacking(b,this->ueX);
    doParsimUnpacking(b,this->ueY);
    doParsimUnpacking(b,this->ueZ);
    doParsimUnpacking(b,this->arriveTime);
    doParsimUnpacking(b,this->timeStamp);
}

int CorePacket::getType() const
{
    return this->type;
}

void CorePacket::setType(int type)
{
    this->type = type;
}

int CorePacket::getBsSimId() const
{
    return this->bsSimId;
}

void CorePacket::setBsSimId(int bsSimId)
{
    this->bsSimId = bsSimId;
}

int CorePacket::getUeSimId() const
{
    return this->ueSimId;
}

void CorePacket::setUeSimId(int ueSimId)
{
    this->ueSimId = ueSimId;
}

double CorePacket::getBsX() const
{
    return this->bsX;
}

void CorePacket::setBsX(double bsX)
{
    this->bsX = bsX;
}

double CorePacket::getBsY() const
{
    return this->bsY;
}

void CorePacket::setBsY(double bsY)
{
    this->bsY = bsY;
}

double CorePacket::getBsZ() const
{
    return this->bsZ;
}

void CorePacket::setBsZ(double bsZ)
{
    this->bsZ = bsZ;
}

double CorePacket::getUeX() const
{
    return this->ueX;
}

void CorePacket::setUeX(double ueX)
{
    this->ueX = ueX;
}

double CorePacket::getUeY() const
{
    return this->ueY;
}

void CorePacket::setUeY(double ueY)
{
    this->ueY = ueY;
}

double CorePacket::getUeZ() const
{
    return this->ueZ;
}

void CorePacket::setUeZ(double ueZ)
{
    this->ueZ = ueZ;
}

double CorePacket::getArriveTime() const
{
    return this->arriveTime;
}

void CorePacket::setArriveTime(double arriveTime)
{
    this->arriveTime = arriveTime;
}

double CorePacket::getTimeStamp() const
{
    return this->timeStamp;
}

void CorePacket::setTimeStamp(double timeStamp)
{
    this->timeStamp = timeStamp;
}

class CorePacketDescriptor : public omnetpp::cClassDescriptor
{
  private:
    mutable const char **propertynames;
  public:
    CorePacketDescriptor();
    virtual ~CorePacketDescriptor();

    virtual bool doesSupport(omnetpp::cObject *obj) const override;
    virtual const char **getPropertyNames() const override;
    virtual const char *getProperty(const char *propertyname) const override;
    virtual int getFieldCount() const override;
    virtual const char *getFieldName(int field) const override;
    virtual int findField(const char *fieldName) const override;
    virtual unsigned int getFieldTypeFlags(int field) const override;
    virtual const char *getFieldTypeString(int field) const override;
    virtual const char **getFieldPropertyNames(int field) const override;
    virtual const char *getFieldProperty(int field, const char *propertyname) const override;
    virtual int getFieldArraySize(void *object, int field) const override;

    virtual const char *getFieldDynamicTypeString(void *object, int field, int i) const override;
    virtual std::string getFieldValueAsString(void *object, int field, int i) const override;
    virtual bool setFieldValueAsString(void *object, int field, int i, const char *value) const override;

    virtual const char *getFieldStructName(int field) const override;
    virtual void *getFieldStructValuePointer(void *object, int field, int i) const override;
};

Register_ClassDescriptor(CorePacketDescriptor)

CorePacketDescriptor::CorePacketDescriptor() : omnetpp::cClassDescriptor("CorePacket", "omnetpp::cPacket")
{
    propertynames = nullptr;
}

CorePacketDescriptor::~CorePacketDescriptor()
{
    delete[] propertynames;
}

bool CorePacketDescriptor::doesSupport(omnetpp::cObject *obj) const
{
    return dynamic_cast<CorePacket *>(obj)!=nullptr;
}

const char **CorePacketDescriptor::getPropertyNames() const
{
    if (!propertynames) {
        static const char *names[] = {  nullptr };
        omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
        const char **basenames = basedesc ? basedesc->getPropertyNames() : nullptr;
        propertynames = mergeLists(basenames, names);
    }
    return propertynames;
}

const char *CorePacketDescriptor::getProperty(const char *propertyname) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    return basedesc ? basedesc->getProperty(propertyname) : nullptr;
}

int CorePacketDescriptor::getFieldCount() const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    return basedesc ? 11+basedesc->getFieldCount() : 11;
}

unsigned int CorePacketDescriptor::getFieldTypeFlags(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldTypeFlags(field);
        field -= basedesc->getFieldCount();
    }
    static unsigned int fieldTypeFlags[] = {
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
        FD_ISEDITABLE,
    };
    return (field>=0 && field<11) ? fieldTypeFlags[field] : 0;
}

const char *CorePacketDescriptor::getFieldName(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldName(field);
        field -= basedesc->getFieldCount();
    }
    static const char *fieldNames[] = {
        "type",
        "bsSimId",
        "ueSimId",
        "bsX",
        "bsY",
        "bsZ",
        "ueX",
        "ueY",
        "ueZ",
        "arriveTime",
        "timeStamp",
    };
    return (field>=0 && field<11) ? fieldNames[field] : nullptr;
}

int CorePacketDescriptor::findField(const char *fieldName) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    int base = basedesc ? basedesc->getFieldCount() : 0;
    if (fieldName[0]=='t' && strcmp(fieldName, "type")==0) return base+0;
    if (fieldName[0]=='b' && strcmp(fieldName, "bsSimId")==0) return base+1;
    if (fieldName[0]=='u' && strcmp(fieldName, "ueSimId")==0) return base+2;
    if (fieldName[0]=='b' && strcmp(fieldName, "bsX")==0) return base+3;
    if (fieldName[0]=='b' && strcmp(fieldName, "bsY")==0) return base+4;
    if (fieldName[0]=='b' && strcmp(fieldName, "bsZ")==0) return base+5;
    if (fieldName[0]=='u' && strcmp(fieldName, "ueX")==0) return base+6;
    if (fieldName[0]=='u' && strcmp(fieldName, "ueY")==0) return base+7;
    if (fieldName[0]=='u' && strcmp(fieldName, "ueZ")==0) return base+8;
    if (fieldName[0]=='a' && strcmp(fieldName, "arriveTime")==0) return base+9;
    if (fieldName[0]=='t' && strcmp(fieldName, "timeStamp")==0) return base+10;
    return basedesc ? basedesc->findField(fieldName) : -1;
}

const char *CorePacketDescriptor::getFieldTypeString(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldTypeString(field);
        field -= basedesc->getFieldCount();
    }
    static const char *fieldTypeStrings[] = {
        "int",
        "int",
        "int",
        "double",
        "double",
        "double",
        "double",
        "double",
        "double",
        "double",
        "double",
    };
    return (field>=0 && field<11) ? fieldTypeStrings[field] : nullptr;
}

const char **CorePacketDescriptor::getFieldPropertyNames(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldPropertyNames(field);
        field -= basedesc->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    }
}

const char *CorePacketDescriptor::getFieldProperty(int field, const char *propertyname) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldProperty(field, propertyname);
        field -= basedesc->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    }
}

int CorePacketDescriptor::getFieldArraySize(void *object, int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldArraySize(object, field);
        field -= basedesc->getFieldCount();
    }
    CorePacket *pp = (CorePacket *)object; (void)pp;
    switch (field) {
        default: return 0;
    }
}

const char *CorePacketDescriptor::getFieldDynamicTypeString(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldDynamicTypeString(object,field,i);
        field -= basedesc->getFieldCount();
    }
    CorePacket *pp = (CorePacket *)object; (void)pp;
    switch (field) {
        default: return nullptr;
    }
}

std::string CorePacketDescriptor::getFieldValueAsString(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldValueAsString(object,field,i);
        field -= basedesc->getFieldCount();
    }
    CorePacket *pp = (CorePacket *)object; (void)pp;
    switch (field) {
        case 0: return long2string(pp->getType());
        case 1: return long2string(pp->getBsSimId());
        case 2: return long2string(pp->getUeSimId());
        case 3: return double2string(pp->getBsX());
        case 4: return double2string(pp->getBsY());
        case 5: return double2string(pp->getBsZ());
        case 6: return double2string(pp->getUeX());
        case 7: return double2string(pp->getUeY());
        case 8: return double2string(pp->getUeZ());
        case 9: return double2string(pp->getArriveTime());
        case 10: return double2string(pp->getTimeStamp());
        default: return "";
    }
}

bool CorePacketDescriptor::setFieldValueAsString(void *object, int field, int i, const char *value) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->setFieldValueAsString(object,field,i,value);
        field -= basedesc->getFieldCount();
    }
    CorePacket *pp = (CorePacket *)object; (void)pp;
    switch (field) {
        case 0: pp->setType(string2long(value)); return true;
        case 1: pp->setBsSimId(string2long(value)); return true;
        case 2: pp->setUeSimId(string2long(value)); return true;
        case 3: pp->setBsX(string2double(value)); return true;
        case 4: pp->setBsY(string2double(value)); return true;
        case 5: pp->setBsZ(string2double(value)); return true;
        case 6: pp->setUeX(string2double(value)); return true;
        case 7: pp->setUeY(string2double(value)); return true;
        case 8: pp->setUeZ(string2double(value)); return true;
        case 9: pp->setArriveTime(string2double(value)); return true;
        case 10: pp->setTimeStamp(string2double(value)); return true;
        default: return false;
    }
}

const char *CorePacketDescriptor::getFieldStructName(int field) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldStructName(field);
        field -= basedesc->getFieldCount();
    }
    switch (field) {
        default: return nullptr;
    };
}

void *CorePacketDescriptor::getFieldStructValuePointer(void *object, int field, int i) const
{
    omnetpp::cClassDescriptor *basedesc = getBaseClassDescriptor();
    if (basedesc) {
        if (field < basedesc->getFieldCount())
            return basedesc->getFieldStructValuePointer(object, field, i);
        field -= basedesc->getFieldCount();
    }
    CorePacket *pp = (CorePacket *)object; (void)pp;
    switch (field) {
        default: return nullptr;
    }
}


