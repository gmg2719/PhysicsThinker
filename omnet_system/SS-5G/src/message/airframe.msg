
message AirFrameMsg
{
    int type;
    int source;
    int destination;
	// Coordinates information
    double x;
    double y;
    double z;
    double destX;
    double destY;
    double destZ;
    double timeStamp;
    // all physical information
    double txPowerUpdate;
    double centerFreq;
}
