
class PubGridMap
{
public:
	// PubGridMap();
	PubGridMap(ros::NodeHandle& nh_, string topic_);
	~PubGridMap();

	void publish(int index_, nav_msgs::OccupancyGrid msg_);

private:
	ros::NodeHandle nh;
	string topic;
	std::vector<std::shared_ptr<ros::Publisher>> pub_grid_map;
};

// PubGridMap::PubGridMap() {}
// PubGridMap::~PubGridMap() {}

PubGridMap::PubGridMap(ros::NodeHandle& nh_, string topic_) 
{
	nh = nh_;
	topic = topic_; 
	pub_grid_map.resize(1e3);
}
PubGridMap::~PubGridMap() {}

void PubGridMap::publish(int index_, nav_msgs::OccupancyGrid msg_) 
{
	if (pub_grid_map[index_] == nullptr) 
	{
		pub_grid_map[index_] = std::make_shared<ros::Publisher>(nh.advertise<nav_msgs::OccupancyGrid>(
		topic+std::to_string(index_), 1));
	}
	pub_grid_map[index_]->publish(msg_);
	// printf("pub %d-th grid map\n", index_);
}